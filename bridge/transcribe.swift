// The smallest bridge that works: a wav path in, words with absolute start and
// end times out as JSON on stdout. Nothing else lives in Swift.
//
//   transcribe <file.wav> [locale] [--install]
//
// --install downloads the language assets for the locale if they are missing.

import Foundation
import Speech
import AVFoundation

struct Word: Codable {
    let text: String
    let start: Double
    let end: Double
}

struct Output: Codable {
    let words: [Word]
    let text: String
    let locale: String
    let seconds: Double
    let audioSeconds: Double
    let installSeconds: Double
    let runs: Int
    let runsWithTime: Int
}

enum BridgeError: Error, CustomStringConvertible {
    case message(String)
    var description: String {
        switch self { case .message(let m): return m }
    }
}

@main
struct Bridge {
    static func main() async {
        let args = CommandLine.arguments
        guard args.count > 1 else {
            FileHandle.standardError.write(
                "usage: transcribe <file.wav> [locale] [--install]\n".data(using: .utf8)!)
            exit(2)
        }
        let path = args[1]
        let localeID = args.count > 2 && !args[2].hasPrefix("--") ? args[2] : "en_US"
        let doInstall = args.contains("--install")

        do {
            let out = try await run(path: path, localeID: localeID, install: doInstall)
            let enc = JSONEncoder()
            enc.outputFormatting = [.withoutEscapingSlashes]
            FileHandle.standardOutput.write(try enc.encode(out))
            FileHandle.standardOutput.write("\n".data(using: .utf8)!)
        } catch {
            FileHandle.standardError.write("error: \(error)\n".data(using: .utf8)!)
            exit(1)
        }
    }

    static func run(path: String, localeID: String, install: Bool) async throws -> Output {
        let url = URL(fileURLWithPath: path)
        let locale = Locale(identifier: localeID)

        let supported = await SpeechTranscriber.supportedLocales
        guard supported.contains(where: { $0.identifier == locale.identifier(.bcp47)
                                       || $0.identifier == localeID
                                       || $0.identifier.replacingOccurrences(of: "_", with: "-")
                                          == localeID.replacingOccurrences(of: "_", with: "-") }) else {
            throw BridgeError.message("locale \(localeID) not supported")
        }

        // audioTimeRange is what puts a start and an end on every run. Without
        // it the result is text with no timing at all.
        let transcriber = SpeechTranscriber(
            locale: locale,
            transcriptionOptions: [],
            reportingOptions: [],
            attributeOptions: [.audioTimeRange])

        var installSeconds = 0.0
        let installed = await SpeechTranscriber.installedLocales
        let haveIt = installed.contains { $0.identifier(.bcp47) == locale.identifier(.bcp47) }
        if !haveIt {
            guard install else {
                throw BridgeError.message(
                    "locale \(localeID) is supported but its assets are not installed; "
                    + "pass --install")
            }
            let t0 = Date()
            if let req = try await AssetInventory.assetInstallationRequest(
                supporting: [transcriber]) {
                try await req.downloadAndInstall()
            }
            installSeconds = Date().timeIntervalSince(t0)
        }

        let analyzer = SpeechAnalyzer(modules: [transcriber])

        let file = try AVAudioFile(forReading: url)
        let audioSeconds = Double(file.length) / file.fileFormat.sampleRate

        let t0 = Date()
        var words: [Word] = []
        var full = ""
        var runs = 0
        var runsWithTime = 0

        let collector = Task {
            for try await result in transcriber.results {
                let attributed = result.text
                for run in attributed.runs {
                    runs += 1
                    let piece = String(attributed[run.range].characters)
                    guard let range = run.audioTimeRange else {
                        full += piece
                        continue
                    }
                    runsWithTime += 1
                    let start = range.start.seconds
                    let end = range.end.seconds
                    full += piece
                    let trimmed = piece.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !trimmed.isEmpty {
                        words.append(Word(text: trimmed, start: start, end: end))
                    }
                }
            }
        }

        if let last = try await analyzer.analyzeSequence(from: file) {
            try await analyzer.finalizeAndFinish(through: last)
        } else {
            try await analyzer.finalizeAndFinishThroughEndOfInput()
        }
        try await collector.value
        let seconds = Date().timeIntervalSince(t0)

        return Output(words: words, text: full, locale: locale.identifier,
                      seconds: seconds, audioSeconds: audioSeconds,
                      installSeconds: installSeconds, runs: runs,
                      runsWithTime: runsWithTime)
    }
}
