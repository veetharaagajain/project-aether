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
        description: "whether one fragment answers the question",
        properties: [
            .init(name: "answers",
                  description: "true only if this fragment states part of the "
                             + "answer to the question",
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
    var seconds: Double = 0
    var perCall: [Double] = []
    var mode: String = ""
    var calls: Int = 0
    var error: String? = nil
    var unavailable: String? = nil
}

let INSTRUCTIONS = """
You decide whether a fragment of someone's recorded speech helps answer a \
question about them.

Answer true only when the fragment states part of the answer. Being about the \
same topic is not enough: a question about what someone ate is not answered by \
them discussing food in general, or by them mentioning a meal without saying \
what it was.

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

    /// One verdict. Independent of every other verdict on purpose.
    static func one(_ question: String, _ fragment: String, _ i: Int)
        async -> (Int, Bool, Double, String?) {
        let t = Date()
        do {
            // A fresh session per candidate. Reusing one would let each verdict
            // see the ones before it, which turns independent classifications
            // into a conversation that can talk itself into a pattern.
            let s = LanguageModelSession(instructions: INSTRUCTIONS)
            let r = try await s.respond(
                to: "Question: \(question)\n\nFragment: \(fragment)\n\n"
                  + "Does this fragment state part of the answer?",
                schema: try verdictSchema(),
                options: GenerationOptions(temperature: 0.0))
            let yes = try r.content.value(Bool.self, forProperty: "answers")
            return (i, yes, Date().timeIntervalSince(t), nil)
        } catch {
            // One candidate failing is not the whole judgement failing, but it
            // must never become a release: an unjudged fragment is withheld and
            // the reason is reported.
            return (i, false, Date().timeIntervalSince(t), "\(error)")
        }
    }

    static func judgeEach(_ req: Request, concurrency: Int) async -> Reply {
        var out = Reply(ok: true, mode: concurrency > 1 ? "each-parallel" : "each")
        var stamps = [Double](repeating: 0, count: req.candidates.count)
        var keep: [Int] = []
        var errs: [String] = []
        let t0 = Date()
        var next = 0
        await withTaskGroup(of: (Int, Bool, Double, String?).self) { group in
            // A bounded window rather than all at once: the model serialises
            // internally, and queueing fifty requests at it buys nothing while
            // making a single slow one hold up the whole batch.
            let width = max(1, min(concurrency, req.candidates.count))
            for _ in 0..<width {
                let i = next; next += 1
                group.addTask { await one(req.question, req.candidates[i], i) }
            }
            while let (i, yes, dt, err) = await group.next() {
                stamps[i] = dt
                if yes { keep.append(i) }
                if let e = err { errs.append("candidate \(i): \(e)") }
                if next < req.candidates.count {
                    let j = next; next += 1
                    group.addTask { await one(req.question, req.candidates[j], j) }
                }
            }
        }
        out.keep = keep.sorted()
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
                to: "Question: x\n\nFragment: y\n\nDoes this fragment state part of the answer?",
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
