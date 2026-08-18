// A long-lived transcription service, so nothing pays process launch or
// framework load per request.
//
// Protocol, line-oriented on stdin, one JSON line back on stdout per request:
//
//   file <locale> <path>\n            transcribe a file
//   pcm <locale> <nbytes>\n           ...followed by exactly nbytes of raw
//                                     little-endian float32 mono 16 kHz PCM
//   quit\n
//
// The pcm form exists for the live path, which holds a segment as an array in
// memory and has no file to point at. Writing it to a temp file and reading it
// back would add two syscalls-worth of disk round trip to a path whose whole
// purpose is latency, and would leave litter to clean up on every crash.

import Foundation
import Speech
import AVFoundation

struct Word: Codable { let text: String; let start: Double; let end: Double }
struct Reply: Codable {
    let ok: Bool
    var words: [Word] = []
    var text: String = ""
    var locale: String = ""
    var seconds: Double = 0
    var audioSeconds: Double = 0
    var runs: Int = 0
    var runsWithTime: Int = 0
    var error: String? = nil
}

/// stdin, read once and buffered here.
///
/// Swift's readLine() keeps its own buffer, so a request whose raw PCM follows
/// the header on the same stream loses those bytes to that buffer and any
/// manual read then blocks forever waiting for data already consumed. Doing
/// all the reading in one place is the only way to mix a text header with a
/// binary payload safely.
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

@main
struct Daemon {
    static let sampleRate = 16000.0

    static func main() async {
        setvbuf(stdout, nil, _IOLBF, 0)
        let enc = JSONEncoder()
        enc.outputFormatting = [.withoutEscapingSlashes]
        let input = Input()
        FileHandle.standardError.write("ready\n".data(using: .utf8)!)

        while let line = input.line() {
            let parts = line.split(separator: " ", maxSplits: 2).map(String.init)
            guard let cmd = parts.first else { continue }
            if cmd == "quit" { break }

            var reply = Reply(ok: false)
            do {
                guard parts.count >= 3 else {
                    throw Err.m("malformed request: \(line)")
                }
                let localeID = parts[1]
                if cmd == "file" {
                    reply = try await transcribe(
                        source: .file(URL(fileURLWithPath: parts[2])),
                        localeID: localeID)
                } else if cmd == "pcm" {
                    guard let n = Int(parts[2]) else { throw Err.m("bad byte count") }
                    guard let data = input.bytes(n) else {
                        throw Err.m("short pcm read")
                    }
                    reply = try await transcribe(source: .pcm(data), localeID: localeID)
                } else {
                    throw Err.m("unknown command: \(cmd)")
                }
            } catch {
                reply = Reply(ok: false, error: "\(error)")
            }
            if let out = try? enc.encode(reply) {
                FileHandle.standardOutput.write(out)
                FileHandle.standardOutput.write("\n".data(using: .utf8)!)
            }
        }
    }

    enum Source { case file(URL); case pcm(Data) }
    enum Err: Error, CustomStringConvertible {
        case m(String)
        var description: String { switch self { case .m(let s): return s } }
    }

    static func transcribe(source: Source, localeID: String) async throws -> Reply {
        let locale = Locale(identifier: localeID)
        let transcriber = SpeechTranscriber(
            locale: locale, transcriptionOptions: [], reportingOptions: [],
            attributeOptions: [.audioTimeRange])

        let installed = await SpeechTranscriber.installedLocales
        guard installed.contains(where: {
            $0.identifier(.bcp47) == locale.identifier(.bcp47)
        }) else {
            throw Err.m("locale \(localeID) is not installed; run transcribe --install")
        }

        let analyzer = SpeechAnalyzer(modules: [transcriber])
        let t0 = Date()
        var words: [Word] = []
        var full = ""
        var runs = 0, timed = 0

        let collector = Task {
            for try await result in transcriber.results {
                let a = result.text
                for run in a.runs {
                    runs += 1
                    let piece = String(a[run.range].characters)
                    full += piece
                    guard let r = run.audioTimeRange else { continue }
                    timed += 1
                    let t = piece.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !t.isEmpty {
                        words.append(Word(text: t, start: r.start.seconds,
                                          end: r.end.seconds))
                    }
                }
            }
        }

        var audioSeconds = 0.0
        switch source {
        case .file(let url):
            let f = try AVAudioFile(forReading: url)
            audioSeconds = Double(f.length) / f.fileFormat.sampleRate
            if let last = try await analyzer.analyzeSequence(from: f) {
                try await analyzer.finalizeAndFinish(through: last)
            } else {
                try await analyzer.finalizeAndFinishThroughEndOfInput()
            }
        case .pcm(let data):
            guard let fmt = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                          sampleRate: sampleRate, channels: 1,
                                          interleaved: false) else {
                throw Err.m("cannot build input format")
            }
            let n = data.count / MemoryLayout<Float>.size
            audioSeconds = Double(n) / sampleRate
            guard let buf = AVAudioPCMBuffer(pcmFormat: fmt,
                                             frameCapacity: AVAudioFrameCount(n)) else {
                throw Err.m("cannot allocate buffer")
            }
            buf.frameLength = AVAudioFrameCount(n)
            data.withUnsafeBytes { raw in
                if let src = raw.baseAddress?.assumingMemoryBound(to: Float.self),
                   let dst = buf.floatChannelData?[0] {
                    dst.update(from: src, count: n)
                }
            }
            let converted: AVAudioPCMBuffer
            if let want = await SpeechAnalyzer.bestAvailableAudioFormat(
                    compatibleWith: [transcriber]),
               want.sampleRate != sampleRate || want.commonFormat != fmt.commonFormat {
                guard let conv = AVAudioConverter(from: fmt, to: want),
                      let out = AVAudioPCMBuffer(
                        pcmFormat: want,
                        frameCapacity: AVAudioFrameCount(
                            Double(n) * want.sampleRate / sampleRate) + 1024)
                else { throw Err.m("cannot convert to analyzer format") }
                var err: NSError?
                var done = false
                conv.convert(to: out, error: &err) { _, status in
                    if done { status.pointee = .noDataNow; return nil }
                    done = true; status.pointee = .haveData; return buf
                }
                if let err { throw Err.m("convert failed: \(err)") }
                converted = out
            } else {
                converted = buf
            }
            let (stream, cont) = AsyncStream<AnalyzerInput>.makeStream()
            try await analyzer.start(inputSequence: stream)
            cont.yield(AnalyzerInput(buffer: converted))
            cont.finish()
            try await analyzer.finalizeAndFinishThroughEndOfInput()
        }

        try await collector.value
        return Reply(ok: true, words: words, text: full,
                     locale: locale.identifier,
                     seconds: Date().timeIntervalSince(t0),
                     audioSeconds: audioSeconds, runs: runs, runsWithTime: timed)
    }
}
