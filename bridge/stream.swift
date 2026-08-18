// Does SpeechAnalyzer stream, and are the times it reports absolute?
//
// Feeds a file in real time through the streaming input, and prints one line
// per result as it arrives: wall clock since start, whether the result is
// volatile (may be revised) or final, the reported audio time range, and the
// text. If the times are chunk-local, as Parakeet's were, they will restart
// from zero on each chunk and this will show it.
//
//   stream <file.wav> [locale] [chunkSeconds]

import Foundation
import Speech
import AVFoundation

@main
struct Streamer {
    static func main() async {
        let args = CommandLine.arguments
        guard args.count > 1 else {
            print("usage: stream <file.wav> [locale] [chunkSeconds]"); exit(2)
        }
        let path = args[1]
        let localeID = args.count > 2 ? args[2] : "en_US"
        let chunk = args.count > 3 ? Double(args[3]) ?? 0.5 : 0.5

        do { try await run(path, localeID, chunk) }
        catch { print("error: \(error)"); exit(1) }
    }

    static func run(_ path: String, _ localeID: String, _ chunk: Double) async throws {
        let transcriber = SpeechTranscriber(
            locale: Locale(identifier: localeID),
            transcriptionOptions: [],
            reportingOptions: [.volatileResults],
            attributeOptions: [.audioTimeRange])
        let analyzer = SpeechAnalyzer(modules: [transcriber])

        let file = try AVAudioFile(forReading: path.hasPrefix("/") ? URL(fileURLWithPath: path)
                                   : URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                                     .appendingPathComponent(path))
        let format = file.processingFormat
        let sr = format.sampleRate
        guard let analyzerFormat = await SpeechAnalyzer.bestAvailableAudioFormat(
                compatibleWith: [transcriber]) else {
            print("no compatible analyzer format"); exit(1)
        }
        print("file format \(sr) Hz, analyzer format \(analyzerFormat.sampleRate) Hz, "
              + "chunk \(chunk)s, volatile results on")

        let converter = AVAudioConverter(from: format, to: analyzerFormat)!
        let (stream, continuation) = AsyncStream<AnalyzerInput>.makeStream()
        try await analyzer.start(inputSequence: stream)

        let t0 = Date()
        let printer = Task {
            for try await result in transcriber.results {
                let now = Date().timeIntervalSince(t0)
                let a = result.text
                var lo = Double.infinity, hi = -Double.infinity
                for run in a.runs {
                    if let r = run.audioTimeRange {
                        lo = min(lo, r.start.seconds); hi = max(hi, r.end.seconds)
                    }
                }
                let kind = result.isFinal ? "FINAL   " : "volatile"
                let text = String(a.characters).trimmingCharacters(in: .whitespacesAndNewlines)
                let range = lo.isFinite ? String(format: "%7.2f-%7.2f", lo, hi) : "   none       "
                print(String(format: "  %6.2fs  %@  audio %@  %@", now, kind, range,
                             text.count > 60 ? String(text.suffix(60)) : text))
            }
        }

        // feed it in real time, chunk by chunk
        let frames = AVAudioFrameCount(chunk * sr)
        let started = Date()
        var fed = 0.0
        while file.framePosition < file.length {
            guard let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames)
            else { break }
            try file.read(into: buf, frameCount: frames)
            if buf.frameLength == 0 { break }
            let outCap = AVAudioFrameCount(Double(buf.frameLength)
                                           * analyzerFormat.sampleRate / sr) + 1024
            guard let out = AVAudioPCMBuffer(pcmFormat: analyzerFormat,
                                             frameCapacity: outCap) else { break }
            var err: NSError?
            var supplied = false
            converter.convert(to: out, error: &err) { _, status in
                if supplied { status.pointee = .noDataNow; return nil }
                supplied = true; status.pointee = .haveData; return buf
            }
            if let err { print("convert error \(err)"); break }
            continuation.yield(AnalyzerInput(buffer: out))
            fed += Double(buf.frameLength) / sr
            let target = started.addingTimeInterval(fed)
            let wait = target.timeIntervalSinceNow
            if wait > 0 { try await Task.sleep(nanoseconds: UInt64(wait * 1e9)) }
        }
        continuation.finish()
        try await analyzer.finalizeAndFinishThroughEndOfInput()
        try await printer.value
        print(String(format: "fed %.1fs of audio in %.1fs wall", fed,
                     Date().timeIntervalSince(started)))
    }
}
