// Does DictationTranscriber produce word timings, and if so how does it compare
// to SpeechTranscriber? Both declare .audioTimeRange in their attribute
// options, so the declaration settles nothing; this runs both and looks at what
// actually comes back on the runs.
//
//   compare <file.wav> [segment.wav] [locale]
//
// Phase 1 prints the declared option surface and locale state for each.
// Phase 2 runs each over the file and, if given, over a short segment the
// length of a typical live utterance, and prints wall clock and the timestamp
// grid. Phase 2 is skipped for an engine that returns no timed runs at all.

import Foundation
import Speech
import AVFoundation

struct Word {
    let text: String
    let start: Double
    let end: Double
}

struct Pass {
    let seconds: Double        // wall clock over the analysis, warmup excluded
    let audioSeconds: Double
    let runs: Int
    let runsWithTime: Int
    let words: [Word]
    let text: String
}

enum Err: Error, CustomStringConvertible {
    case m(String)
    var description: String { switch self { case .m(let s): return s } }
}

func audioSeconds(_ url: URL) throws -> Double {
    let f = try AVAudioFile(forReading: url)
    return Double(f.length) / f.fileFormat.sampleRate
}

// --- the two runners --------------------------------------------------------
// Near-identical on purpose. The result types are distinct and unrelated, and
// one generic function over them would be longer than both of these.

func runSpeech(_ url: URL, _ locale: Locale) async throws -> Pass {
    let t = SpeechTranscriber(locale: locale, transcriptionOptions: [],
                              reportingOptions: [], attributeOptions: [.audioTimeRange])
    let analyzer = SpeechAnalyzer(modules: [t])
    let file = try AVAudioFile(forReading: url)
    let audio = Double(file.length) / file.fileFormat.sampleRate

    var words: [Word] = [], full = ""
    var runs = 0, timed = 0
    let collector = Task {
        for try await result in t.results {
            let a = result.text
            for run in a.runs {
                runs += 1
                let piece = String(a[run.range].characters)
                full += piece
                guard let r = run.audioTimeRange else { continue }
                timed += 1
                let trimmed = piece.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    words.append(Word(text: trimmed, start: r.start.seconds, end: r.end.seconds))
                }
            }
        }
    }
    let t0 = Date()
    if let last = try await analyzer.analyzeSequence(from: file) {
        try await analyzer.finalizeAndFinish(through: last)
    } else {
        try await analyzer.finalizeAndFinishThroughEndOfInput()
    }
    try await collector.value
    let secs = Date().timeIntervalSince(t0)
    return Pass(seconds: secs, audioSeconds: audio, runs: runs,
                runsWithTime: timed, words: words, text: full)
}

func runDictation(_ url: URL, _ locale: Locale) async throws -> Pass {
    let t = DictationTranscriber(locale: locale, contentHints: [],
                                 transcriptionOptions: [], reportingOptions: [],
                                 attributeOptions: [.audioTimeRange])
    let analyzer = SpeechAnalyzer(modules: [t])
    let file = try AVAudioFile(forReading: url)
    let audio = Double(file.length) / file.fileFormat.sampleRate

    var words: [Word] = [], full = ""
    var runs = 0, timed = 0
    let collector = Task {
        for try await result in t.results {
            let a = result.text
            for run in a.runs {
                runs += 1
                let piece = String(a[run.range].characters)
                full += piece
                guard let r = run.audioTimeRange else { continue }
                timed += 1
                let trimmed = piece.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    words.append(Word(text: trimmed, start: r.start.seconds, end: r.end.seconds))
                }
            }
        }
    }
    let t0 = Date()
    if let last = try await analyzer.analyzeSequence(from: file) {
        try await analyzer.finalizeAndFinish(through: last)
    } else {
        try await analyzer.finalizeAndFinishThroughEndOfInput()
    }
    try await collector.value
    let secs = Date().timeIntervalSince(t0)
    return Pass(seconds: secs, audioSeconds: audio, runs: runs,
                runsWithTime: timed, words: words, text: full)
}

// --- reporting --------------------------------------------------------------

func report(_ name: String, _ label: String, _ p: Pass) {
    let rtf = p.seconds / max(p.audioSeconds, 0.0001)
    print(String(format: "  %@ / %@: %.3f s wall over %.2f s audio  (%.3fx realtime)",
                 name, label, p.seconds, p.audioSeconds, rtf))
    print("    runs \(p.runs), runs carrying audioTimeRange \(p.runsWithTime), words \(p.words.count)")
}

func grid(_ name: String, _ p: Pass) {
    print("\n  \(name) timestamp grid (\(p.words.count) words):")
    if p.words.isEmpty {
        print("    (none — no run carried an audioTimeRange)")
        print("    text: \(p.text.trimmingCharacters(in: .whitespacesAndNewlines))")
        return
    }
    var prevEnd: Double? = nil
    for w in p.words {
        let gap = prevEnd.map { String(format: "%+.3f", w.start - $0) } ?? "     ."
        print(String(format: "    %8.3f %8.3f  %6.3f  %@  %@",
                     w.start, w.end, w.end - w.start, gap, w.text))
        prevEnd = w.end
    }
    // The grid question: are these real per-word boundaries or a quantised
    // lattice? Distinct start values and their smallest nonzero difference say
    // which.
    let starts = Set(p.words.map { ($0.start * 1000).rounded() / 1000 })
    var minStep = Double.greatestFiniteMagnitude
    let sorted = p.words.map { $0.start }.sorted()
    for i in 1..<max(sorted.count, 2) where i < sorted.count {
        let d = sorted[i] - sorted[i-1]
        if d > 1e-9 { minStep = min(minStep, d) }
    }
    print(String(format: "    distinct starts %d of %d, smallest nonzero step %.4f s",
                 starts.count, p.words.count,
                 minStep == .greatestFiniteMagnitude ? 0 : minStep))
}

@main
struct Compare {
    static func main() async {
        let args = CommandLine.arguments
        guard args.count > 1 else {
            print("usage: compare <file.wav> [segment.wav] [locale]"); exit(2)
        }
        let fileURL = URL(fileURLWithPath: args[1])
        let segURL: URL? = args.count > 2 && !args[2].hasPrefix("--")
            ? URL(fileURLWithPath: args[2]) : nil
        let localeID = args.count > 3 ? args[3] : "en_US"
        let locale = Locale(identifier: localeID)

        // --- phase 1: what each one says it can do --------------------------
        print("=== phase 1: declared surface ===")
        print("locale requested: \(localeID)")

        print("\nSpeechTranscriber")
        print("  ResultAttributeOption.allCases: \(SpeechTranscriber.ResultAttributeOption.allCases)")
        print("  TranscriptionOption.allCases:   \(SpeechTranscriber.TranscriptionOption.allCases)")
        print("  ReportingOption.allCases:       \(SpeechTranscriber.ReportingOption.allCases)")
        let sSup = await SpeechTranscriber.supportedLocales
        let sInst = await SpeechTranscriber.installedLocales
        print("  supported \(sSup.count), installed \(sInst.map { $0.identifier(.bcp47) }.sorted())")

        print("\nDictationTranscriber")
        print("  ResultAttributeOption.allCases: \(DictationTranscriber.ResultAttributeOption.allCases)")
        print("  TranscriptionOption.allCases:   \(DictationTranscriber.TranscriptionOption.allCases)")
        print("  ReportingOption.allCases:       \(DictationTranscriber.ReportingOption.allCases)")
        let dSup = await DictationTranscriber.supportedLocales
        let dInst = await DictationTranscriber.installedLocales
        print("  supported \(dSup.count), installed \(dInst.map { $0.identifier(.bcp47) }.sorted())")

        let want = locale.identifier(.bcp47)
        let sHave = sInst.contains { $0.identifier(.bcp47) == want }
        let dHave = dInst.contains { $0.identifier(.bcp47) == want }
        print("\n  \(want) installed for SpeechTranscriber: \(sHave), for DictationTranscriber: \(dHave)")
        if !sHave || !dHave {
            print("  (a missing locale means that engine cannot run below; install it first)")
        }

        // --- phase 2: what each one actually emits --------------------------
        print("\n=== phase 2: does audioTimeRange actually arrive? ===")
        let probe = segURL ?? fileURL
        var sProbe: Pass? = nil, dProbe: Pass? = nil
        do { sProbe = try await runSpeech(probe, locale) }
        catch { print("  SpeechTranscriber failed: \(error)") }
        do { dProbe = try await runDictation(probe, locale) }
        catch { print("  DictationTranscriber failed: \(error)") }

        let sTimed = (sProbe?.runsWithTime ?? 0) > 0
        let dTimed = (dProbe?.runsWithTime ?? 0) > 0
        print("  SpeechTranscriber timed runs: \(sProbe?.runsWithTime ?? 0) of \(sProbe?.runs ?? 0)")
        print("  DictationTranscriber timed runs: \(dProbe?.runsWithTime ?? 0) of \(dProbe?.runs ?? 0)")

        guard dTimed else {
            print("\n  DictationTranscriber returned no run carrying an audioTimeRange.")
            print("  That settles it: it cannot give this project per-word timings.")
            if let d = dProbe {
                print("  text it did return: \(d.text.trimmingCharacters(in: .whitespacesAndNewlines))")
            }
            exit(0)
        }
        guard sTimed else {
            print("\n  SpeechTranscriber returned no timed runs, which is unexpected;")
            print("  the comparison below would not mean anything. Stopping.")
            exit(1)
        }

        // Those probe runs were also the warmup: assets and the analyzer are
        // loaded now, so the timings below are analysis, not first-use cost.
        print("\n=== phase 3: wall clock (warm) ===")
        for (label, url) in [("full file", fileURL)] + (segURL.map { [("live-length segment", $0)] } ?? []) {
            print("\n\(label): \(url.lastPathComponent)")
            for pass in 1...3 {
                do {
                    let s = try await runSpeech(url, locale)
                    let d = try await runDictation(url, locale)
                    report("SpeechTranscriber   ", "pass \(pass)", s)
                    report("DictationTranscriber", "pass \(pass)", d)
                    if pass == 3 {
                        grid("SpeechTranscriber", s)
                        grid("DictationTranscriber", d)
                    }
                } catch {
                    print("  pass \(pass) failed: \(error)")
                }
            }
        }
    }
}
