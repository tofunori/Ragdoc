import SwiftUI

struct SettingsView: View {
    @AppStorage("converterPath") private var converterPath = PipelineConfiguration.defaultConverterPath
    @AppStorage("nasHost") private var nasHost = "rorqual"
    @AppStorage("remoteRoot") private var remoteRoot = "/volume1/Services/mcp/ragdoc"

    var body: some View {
        Form {
            Section("MinerU") {
                TextField("Convertisseur", text: $converterPath)
                LabeledContent("Jeton") {
                    Label(tokenExists ? "Détecté" : "Absent", systemImage: tokenExists ? "checkmark.circle" : "exclamationmark.triangle")
                        .foregroundStyle(tokenExists ? Color.green : Color.orange)
                }
            }
            Section("Ragdoc sur le NAS") {
                TextField("Hôte SSH", text: $nasHost)
                TextField("Dossier", text: $remoteRoot)
            }
        }
        .formStyle(.grouped)
        .frame(width: 520, height: 280)
        .scenePadding()
    }

    private var tokenExists: Bool {
        FileManager.default.fileExists(
            atPath: FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent(".mineru_token").path
        )
    }
}
