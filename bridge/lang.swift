import Foundation
import Speech

@main
struct Lang {
    static func main() async {
        // 1. can a transcriber be built for Kannada, and what do assets cost?
        let kn = Locale(identifier: "kn_IN")
        let t = SpeechTranscriber(locale: kn, transcriptionOptions: [],
                                  reportingOptions: [], attributeOptions: [.audioTimeRange])
        let installed = await SpeechTranscriber.installedLocales
        print("kn_IN installed already: \(installed.contains { $0.identifier(.bcp47) == kn.identifier(.bcp47) })")
        do {
            if let req = try await AssetInventory.assetInstallationRequest(supporting: [t]) {
                print("kn_IN needs an install. downloading...")
                let t0 = Date()
                try await req.downloadAndInstall()
                print(String(format: "kn_IN installed in %.1fs", Date().timeIntervalSince(t0)))
            } else {
                print("kn_IN: no installation needed")
            }
        } catch { print("kn_IN install failed: \(error)") }

        // 2. what does a transcriber accept? one locale, or several?
        print("reserved locales: \(await AssetInventory.reservedLocales.map { $0.identifier })")
        print("maximum reserved: \(AssetInventory.maximumReservedLocales)")

        // 3. is there a multi-language locale, and does it build?
        let mul = Locale(identifier: "mul_IN")
        let mt = SpeechTranscriber(locale: mul, transcriptionOptions: [],
                                   reportingOptions: [], attributeOptions: [.audioTimeRange])
        do {
            if let req = try await AssetInventory.assetInstallationRequest(supporting: [mt]) {
                print("mul_IN needs an install. downloading...")
                let t0 = Date()
                try await req.downloadAndInstall()
                print(String(format: "mul_IN installed in %.1fs", Date().timeIntervalSince(t0)))
            } else { print("mul_IN: no installation needed") }
        } catch { print("mul_IN install failed: \(error)") }

        let now = await SpeechTranscriber.installedLocales
        print("installed now: \(now.map { $0.identifier }.sorted())")
    }
}
