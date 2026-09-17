import SwiftUI

struct SettingsView: View {
    @AppStorage("converterPath") private var converterPath = PipelineConfiguration.defaultConverterPath
    @AppStorage("nasHost") private var nasHost = "rorqual"
    @AppStorage("remoteRoot") private var remoteRoot = "/volume1/Services/mcp/ragdoc"
    @State private var newAPIKey = ""
    @State private var credentialConfigured = MistralCredentialStore.isConfigured
    @State private var credentialMessage: String?

    var body: some View {
        Form {
            Section("Conversion du PDF") {
                Picker("Service", selection: providerBinding) {
                    ForEach(PipelineConfiguration.ConverterKind.allCases) { provider in
                        Text(provider.title).tag(provider)
                    }
                }
                Text("Mistral OCR est le service principal. MinerU reste disponible comme solution de secours.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Section("Mistral OCR") {
                LabeledContent("Clé API") {
                    Label(
                        credentialConfigured ? "Configurée" : "Absente",
                        systemImage: credentialConfigured ? "checkmark.circle" : "exclamationmark.triangle"
                    )
                    .foregroundStyle(credentialConfigured ? Color.green : Color.orange)
                }
                SecureField("Nouvelle clé Mistral", text: $newAPIKey)
                    .textFieldStyle(.roundedBorder)
                HStack {
                    Link("Créer une clé…", destination: URL(string: "https://console.mistral.ai/api-keys")!)
                    Spacer()
                    Button("Supprimer") { deleteCredential() }
                        .disabled(!MistralCredentialStore.hasKeychainCredential)
                    Button("Enregistrer") { saveCredential() }
                        .buttonStyle(.borderedProminent)
                        .disabled(newAPIKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
                if let credentialMessage {
                    Text(credentialMessage).font(.caption).foregroundStyle(.secondary)
                }
            }
            Section("MinerU — secours") {
                LabeledContent("Jeton") {
                    Label(tokenExists ? "Détecté" : "Absent", systemImage: tokenExists ? "checkmark.circle" : "exclamationmark.triangle")
                        .foregroundStyle(tokenExists ? Color.green : Color.orange)
                }
            }
            DisclosureGroup("Réglages avancés") {
                TextField("Convertisseur", text: $converterPath)
            }
            Section("Ragdoc sur le NAS") {
                TextField("Hôte SSH", text: $nasHost)
                TextField("Dossier", text: $remoteRoot)
            }
        }
        .formStyle(.grouped)
        .frame(width: 560, height: 520)
        .scenePadding()
        .onAppear {
            let current = PipelineConfiguration.current()
            converterPath = current.converterPath
            credentialConfigured = MistralCredentialStore.isConfigured
        }
    }

    private var providerBinding: Binding<PipelineConfiguration.ConverterKind> {
        Binding {
            PipelineConfiguration.converterKind(for: converterPath)
        } set: { provider in
            switch provider {
            case .mistral: converterPath = PipelineConfiguration.defaultMistralConverterPath
            case .mineru: converterPath = PipelineConfiguration.defaultMinerUConverterPath
            case .custom: break
            }
        }
    }

    private func saveCredential() {
        do {
            try MistralCredentialStore.save(newAPIKey)
            newAPIKey = ""
            credentialConfigured = true
            credentialMessage = "Clé enregistrée dans le trousseau macOS."
        } catch {
            credentialMessage = error.localizedDescription
        }
    }

    private func deleteCredential() {
        do {
            try MistralCredentialStore.delete()
            credentialConfigured = MistralCredentialStore.isConfigured
            credentialMessage = credentialConfigured
                ? "Une clé externe est encore configurée."
                : "Clé supprimée du trousseau macOS."
        } catch {
            credentialMessage = error.localizedDescription
        }
    }

    private var tokenExists: Bool {
        FileManager.default.fileExists(
            atPath: FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent(".mineru_token").path
        )
    }
}
