import Foundation
import Speech

@main
struct Probe {
    static func main() async {
        print("SpeechTranscriber supported locales:")
        let sup = await SpeechTranscriber.supportedLocales
        for l in sup.sorted(by: { $0.identifier < $1.identifier }) {
            print("  \(l.identifier)  \(l.localizedString(forIdentifier: l.identifier) ?? "")")
        }
        print("count: \(sup.count)")
        let inst = await SpeechTranscriber.installedLocales
        print("installed: \(inst.map { $0.identifier }.sorted())")
        let kn = sup.filter { $0.identifier.hasPrefix("kn") }
        print("Kannada supported: \(kn.map { $0.identifier })")
    }
}
