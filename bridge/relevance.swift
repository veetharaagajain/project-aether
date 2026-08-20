// The first piece of Aether that decides rather than measures.
//
// One job: given a question and the fragments memory returned for it, say
// which of them actually answer it. Not whether the question was addressed to
// Aether, not which model to consult, not whether to speak.
//
// Apple's on-device model, through FoundationModels. Chosen because it is free,
// unbounded, and runs here -- a relevance judgement sees the person's speech,
// so a judge that needed a network call would undo the property the gate
// exists to provide. Apple describes this model as suited to classification
// and extraction rather than reasoning, which is what asking "does this
// fragment answer this question" is.
//
// Long-lived, like bridge/daemon.swift, so nothing pays framework load or
// model warm-up per request.
//
// Protocol, line-oriented on stdin, one JSON line back on stdout per request:
//
//   ready\n                    is the model available at all
//   judge <nbytes>\n           ...followed by exactly nbytes of JSON:
//                              {"question": "...", "candidates": ["...", ...],
//                               "mode": "each" | "batch"}
//   quit\n
//
// Two modes, because which one suits the model is a question rather than an
// assumption. "each" asks a separate yes/no per candidate, which is pure
// classification and what Apple says the model is for. "batch" shows it all
// the candidates at once and asks which answer, which is one call but closer
// to ranking. relevance.py measures both.

import Foundation
import FoundationModels

// The shape the model must fill in. Guided generation rather than parsing
// prose: a judge whose output needs a regular expression is a judge that fails
// open the first time it phrases itself differently.
//
// Built with DynamicGenerationSchema rather than the @Generable macro. The
// macro needs the FoundationModelsMacros compiler plugin, which ships with
// Xcode and not with the Command Line Tools this machine has, so the macro
// form will not build here at all. The dynamic form is the same guarantee
// expressed at runtime, and it has the incidental advantage of not requiring a
// full Xcode install to rebuild the bridge.
func verdictSchema() throws -> GenerationSchema {
    let root = DynamicGenerationSchema(
        name: "Verdict",
        description: "whether the marked fragment answers the question",
        properties: [
            .init(name: "answers",
                  description: "true only if the marked line states part of the "
                             + "answer to the question, reading it in context",
                  schema: DynamicGenerationSchema(type: Bool.self)),
            // Asked in the same call rather than a second one. It costs nothing
            // extra and it is what decides whether the neighbours are released
            // alongside the fragment or only used to understand it.
            .init(name: "needsContext",
                  description: "true if the marked line cannot be understood on "
                             + "its own -- for example it continues the sentence "
                             + "above it, or its subject is only named in a "
                             + "surrounding line",
                  schema: DynamicGenerationSchema(type: Bool.self))
        ])
    return try GenerationSchema(root: root, dependencies: [])
}

func picksSchema() throws -> GenerationSchema {
    let root = DynamicGenerationSchema(
        name: "Picks",
        description: "which fragments answer the question",
        properties: [
            .init(name: "numbers",
                  description: "the 1-based numbers of only those fragments "
                             + "that state part of the answer; empty if none do",
                  schema: DynamicGenerationSchema(
                      arrayOf: DynamicGenerationSchema(type: Int.self)))
        ])
    return try GenerationSchema(root: root, dependencies: [])
}

struct Request: Codable {
    let question: String
    let candidates: [String]
    var mode: String = "each"
    var concurrency: Int = 4
}

struct Reply: Codable {
    var ok: Bool
    var keep: [Int] = []          // indices into candidates, 0-based
    var needsContext: [Int] = []  // ...of which these do not stand alone
    var seconds: Double = 0
    var perCall: [Double] = []
    var mode: String = ""
    var calls: Int = 0
    var error: String? = nil
    var unavailable: String? = nil
}

let PER_CALL_TIMEOUT_S = 12.0

let INSTRUCTIONS = """
You decide whether a fragment of someone's recorded speech helps answer a \
question about them.

You are shown a short stretch of transcript. One line is marked with >>. Judge \
only that line. The unmarked lines are what was said immediately before and \
after it, and they are there so you can understand the marked line -- they are \
not what you are judging.

Set answers to true only when the marked line states part of the answer, read \
in that context. Speech is often split mid-sentence, so a line that continues \
the sentence above it may well state the answer even though it reads as a \
fragment on its own. Being about the same topic is still not enough: a question \
about what someone ate is not answered by them discussing food in general, or \
by them mentioning a meal without saying what it was.

Set needsContext to true when the marked line cannot be understood alone -- it \
continues a neighbouring sentence, or its subject is only named nearby.

Answer false when you are unsure. Withholding a fragment loses an answer; \
releasing one hands over private speech that was not asked for.
"""

/// stdin, read once and buffered here. The same reader bridge/daemon.swift
/// uses, and for the same reason: a request whose JSON body follows its header
/// on one stream loses those bytes to readLine's own buffer, and any manual
/// read then blocks forever waiting for data already consumed.
final class Input {
    private var buf = Data()
    private let handle = FileHandle.standardInput

    private func fill() -> Bool {
        let chunk = handle.availableData
        if chunk.isEmpty { return false }
        buf.append(chunk)
        return true
    }

    func line() -> String? {
        while true {
            if let i = buf.firstIndex(of: 0x0A) {
                let s = String(data: buf[buf.startIndex..<i], encoding: .utf8)
                buf.removeSubrange(buf.startIndex...i)
                return s
            }
            if !fill() { return buf.isEmpty ? nil : nil }
        }
    }

    func bytes(_ n: Int) -> Data? {
        while buf.count < n {
            if !fill() { return nil }
        }
        let out = buf.prefix(n)
        buf.removeSubrange(buf.startIndex..<(buf.startIndex + n))
        return Data(out)
    }
}


func firstLine(_ s: String) -> String {
    s.split(separator: "\n").first.map(String.init) ?? s
}

@main
struct Bridge {
    static var session: LanguageModelSession?

    static func available() -> String? {
        switch SystemLanguageModel.default.availability {
        case .available: return nil
        case .unavailable(let reason): return "\(reason)"
        @unknown default: return "unknown availability"
        }
    }

    /// One verdict, or nothing if the model stops responding.
    ///
    /// Some prompts wedge the on-device model: it returns neither an answer nor
    /// an error and the call never completes. A five-line window over this
    /// corpus reproduced it, while the same window at four lines answered in
    /// 0.6 s. Without a bound, one such prompt blocks the whole judgement and
    /// therefore the release that waits on it, indefinitely.
    ///
    /// A timeout here rather than only in the caller, so one wedged candidate
    /// costs that candidate and not the batch. It fails the same way every
    /// other error does: withheld, and reported.
    static func one(_ question: String, _ fragment: String, _ i: Int)
        async -> (Int, Bool, Bool, Double, String?) {
        let t = Date()
        return await withTaskGroup(of: (Int, Bool, Bool, Double, String?)?.self) { g in
            g.addTask { await judgeOne(question, fragment, i, t) }
            g.addTask {
                try? await Task.sleep(nanoseconds: UInt64(PER_CALL_TIMEOUT_S * 1e9))
                return nil
            }
            let first = await g.next() ?? nil
            g.cancelAll()
            return first ?? (i, false, false, Date().timeIntervalSince(t),
                             "timed out after \(PER_CALL_TIMEOUT_S)s")
        }
    }

    static func judgeOne(_ question: String, _ fragment: String, _ i: Int,
                         _ t: Date) async -> (Int, Bool, Bool, Double, String?) {
        do {
            // A fresh session per candidate. Reusing one would let each verdict
            // see the ones before it, which turns independent classifications
            // into a conversation that can talk itself into a pattern.
            let s = LanguageModelSession(instructions: INSTRUCTIONS)
            let r = try await s.respond(
                to: "Question: \(question)\n\nTranscript:\n\(fragment)\n\n"
                  + "Does the marked line state part of the answer?",
                schema: try verdictSchema(),
                options: GenerationOptions(temperature: 0.0))
            let yes = try r.content.value(Bool.self, forProperty: "answers")
            let needs = (try? r.content.value(Bool.self,
                                              forProperty: "needsContext")) ?? false
            return (i, yes, needs, Date().timeIntervalSince(t), nil)
        } catch {
            // One candidate failing is not the whole judgement failing, but it
            // must never become a release: an unjudged fragment is withheld and
            // the reason is reported.
            return (i, false, false, Date().timeIntervalSince(t), "\(error)")
        }
    }

    static func judgeEach(_ req: Request, concurrency: Int) async -> Reply {
        var out = Reply(ok: true, mode: concurrency > 1 ? "each-parallel" : "each")
        var stamps = [Double](repeating: 0, count: req.candidates.count)
        var keep: [Int] = []
        var needsCtx: [Int] = []
        var errs: [String] = []
        let t0 = Date()
        var next = 0
        await withTaskGroup(of: (Int, Bool, Bool, Double, String?).self) { group in
            // A bounded window rather than all at once: the model serialises
            // internally, and queueing fifty requests at it buys nothing while
            // making a single slow one hold up the whole batch.
            let width = max(1, min(concurrency, req.candidates.count))
            for _ in 0..<width {
                let i = next; next += 1
                group.addTask { await one(req.question, req.candidates[i], i) }
            }
            while let (i, yes, needs, dt, err) = await group.next() {
                stamps[i] = dt
                if yes { keep.append(i) }
                if needs { needsCtx.append(i) }
                if let e = err { errs.append("candidate \(i): \(e)") }
                if next < req.candidates.count {
                    let j = next; next += 1
                    group.addTask { await one(req.question, req.candidates[j], j) }
                }
            }
        }
        out.keep = keep.sorted()
        out.needsContext = needsCtx.sorted()
        out.perCall = stamps
        out.calls = req.candidates.count
        out.error = errs.isEmpty ? nil : errs.joined(separator: "; ")
        out.seconds = Date().timeIntervalSince(t0)
        return out
    }

    static func judgeBatch(_ req: Request) async -> Reply {
        var out = Reply(ok: true, mode: "batch")
        let t0 = Date()
        let listed = req.candidates.enumerated()
            .map { "\($0.offset + 1). \($0.element)" }.joined(separator: "\n")
        do {
            let s = LanguageModelSession(instructions: INSTRUCTIONS)
            let r = try await s.respond(
                to: "Question: \(req.question)\n\nFragments:\n\(listed)\n\n"
                  + "Which fragments state part of the answer? Give only their "
                  + "numbers, and none if no fragment does.",
                schema: try picksSchema(),
                options: GenerationOptions(temperature: 0.0))
            let nums = try r.content.value([Int].self, forProperty: "numbers")
            // 1-based in, 0-based out, and anything out of range dropped
            // rather than trusted
            out.keep = nums
                .map { $0 - 1 }
                .filter { $0 >= 0 && $0 < req.candidates.count }
                .reduce(into: [Int]()) { if !$0.contains($1) { $0.append($1) } }
                .sorted()
        } catch {
            out.ok = false
            out.error = "\(error)"
        }
        out.calls = 1
        out.seconds = Date().timeIntervalSince(t0)
        out.perCall = [out.seconds]
        return out
    }

    static func write(_ r: Reply) {
        let enc = JSONEncoder()
        enc.outputFormatting = [.withoutEscapingSlashes]
        if let d = try? enc.encode(r) {
            FileHandle.standardOutput.write(d)
            FileHandle.standardOutput.write("\n".data(using: .utf8)!)
        }
    }

    static func main() async {
        let input = Input()
        if let why = available() {
            FileHandle.standardError.write(
                "unavailable: \(why)\n".data(using: .utf8)!)
        } else {
            // warm the model once so the first real request does not pay for it
            let s = LanguageModelSession(instructions: INSTRUCTIONS)
            _ = try? await s.respond(
                to: "Question: x\n\nTranscript:\n>> y\n\nDoes the marked line state part of the answer?",
                schema: try! verdictSchema(),
                options: GenerationOptions(temperature: 0.0))
            session = s
        }
        FileHandle.standardError.write("ready\n".data(using: .utf8)!)

        while let line = input.line() {
            let parts = line.split(separator: " ").map(String.init)
            guard let cmd = parts.first else { continue }
            if cmd == "quit" { break }
            if cmd == "ready" {
                var r = Reply(ok: available() == nil)
                r.unavailable = available()
                write(r)
                continue
            }
            guard cmd == "judge", parts.count > 1, let n = Int(parts[1]),
                  let body = input.bytes(n) else {
                write(Reply(ok: false, error: "bad request: \(line)"))
                continue
            }
            guard let req = try? JSONDecoder().decode(Request.self, from: body) else {
                write(Reply(ok: false, error: "bad json"))
                continue
            }
            if let why = available() {
                var r = Reply(ok: false)
                r.unavailable = why
                r.error = "model unavailable"
                write(r)
                continue
            }
            let reply = req.mode == "batch"
                ? await judgeBatch(req)
                : await judgeEach(req, concurrency: req.concurrency)
            write(reply)
        }
    }
}
