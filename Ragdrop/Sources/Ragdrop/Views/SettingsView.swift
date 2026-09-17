import SwiftUI

struct SettingsView: View {
    @Bindable var monitor: ZoteroMonitorStore
    @AppStorage("converterPath") private var converterPath = PipelineConfiguration.defaultConverterPath
    @AppStorage("nasHost") private var nasHost = "ragdoc-server"
    @AppStorage("remoteRoot") private var remoteRoot = "/srv/ragdoc"
    @State private var newAPIKey = ""
    @AppStorage("appearanceMode", store: RagdropAppearance.defaults) private var appearanceMode = RagdropAppearance.light.rawValue
    let isIsolated: Bool
    @State private var credentialConfigured = false
    @State private var credentialMessage: String?

    init(isIsolated: Bool = false, monitor: ZoteroMonitorStore? = nil) {
        self.monitor = monitor ?? ZoteroMonitorStore(isIsolated: isIsolated)
        self.isIsolated = isIsolated
        let defaults = isIsolated ? UserDefaults(suiteName: "com.tofunori.ragdrop.theme-demo")! : .standard
        _converterPath = AppStorage(wrappedValue: PipelineConfiguration.defaultConverterPath, "converterPath", store: defaults)
        _nasHost = AppStorage(wrappedValue: "ragdoc-server", "nasHost", store: defaults)
        _remoteRoot = AppStorage(wrappedValue: "/srv/ragdoc", "remoteRoot", store: defaults)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            PageHeading(title: "Réglages", subtitle: "Conversion, accès et destination de vos articles.")
            if isIsolated { InlineNotice(text: "Validation isolée · aucune clé réelle n’est lue ou modifiée.") }
            Form {
            Section("Apparence") {
                Picker("Thème", selection: $appearanceMode) {
                    ForEach(RagdropAppearance.allCases) { appearance in
                        Text(appearance.title).tag(appearance.rawValue)
                    }
                }.pickerStyle(.segmented)
            }

            Section("Nouveaux articles Zotero") {
                Toggle("Surveiller Zotero", isOn: Binding(get: { monitor.isEnabled }, set: { monitor.setEnabled($0) }))
                Text("Signaler les nouveaux PDF locaux, sans import automatique.")
                    .font(.callout).foregroundStyle(RagdropTheme.secondary)
                DisclosureGroup("Fonctionnement") {
                    Text("Pendant que Ragdrop fonctionne, vérification toutes les 5 minutes via l’API locale de Zotero. Zotero doit être ouverte.")
                    Text("À l’activation, les PDF déjà présents servent de référence et ne déclenchent pas d’alerte. Utilisez Depuis Zotero pour les examiner. Les éléments sans PDF local seront pris en compte lorsqu’un PDF sera disponible.")
                }.font(.callout).foregroundStyle(RagdropTheme.secondary)
                if monitor.isEnabled {
                    Text(monitor.statusText).font(.caption).foregroundStyle(RagdropTheme.secondary)
                    if let error = monitor.errorMessage { Text(error).font(.caption).foregroundStyle(RagdropTheme.warning) }
                    HStack {
                        if let date = monitor.lastLocalCheck { Text("Dernière lecture : \(date.formatted(date: .omitted, time: .shortened))").font(.caption) }
                        Spacer()
                        Button("Vérifier maintenant") { monitor.checkNow() }.disabled(monitor.isChecking || isIsolated)
                    }
                }
            }

            Section("Conversion du PDF") {
                Picker("Service", selection: providerBinding) {
                    ForEach(PipelineConfiguration.ConverterKind.allCases) { provider in
                        Text(provider.title).tag(provider)
                    }
                }
                Text("Mistral OCR est le service principal. MinerU reste disponible comme solution de secours.")
                    .font(.caption)
                    .foregroundStyle(RagdropTheme.secondary)
            }
            Section("Mistral OCR") {
                LabeledContent("Clé API") {
                    Label(
                        credentialConfigured ? "Configurée" : "Absente",
                        systemImage: credentialConfigured ? "checkmark.circle" : "exclamationmark.triangle"
                    )
                    .foregroundStyle(credentialConfigured ? RagdropTheme.success : RagdropTheme.warning)
                }
                SecureField("Nouvelle clé Mistral", text: $newAPIKey)
                    .textFieldStyle(.roundedBorder)
                HStack {
                    Link("Créer une clé…", destination: URL(string: "https://console.mistral.ai/api-keys")!)
                    Spacer()
                    Button("Supprimer") { deleteCredential() }
                        .disabled(isIsolated || !hasStoredCredential)
                    Button("Enregistrer") { saveCredential() }
                        .buttonStyle(RagdropPrimaryButtonStyle())
                        .disabled(newAPIKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
                if let credentialMessage {
                    Text(credentialMessage).font(.caption).foregroundStyle(RagdropTheme.secondary)
                }
            }
            Section("MinerU — secours") {
                LabeledContent("Jeton") {
                    Label(tokenExists ? "Détecté" : "Absent", systemImage: tokenExists ? "checkmark.circle" : "exclamationmark.triangle")
                        .foregroundStyle(tokenExists ? RagdropTheme.success : RagdropTheme.warning)
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
        .scrollContentBackground(.hidden)
        }
        .padding(RagdropTheme.pagePadding)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .ragdropSurface()
        .onAppear {
            guard !isIsolated else { return }
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
        guard !isIsolated else { newAPIKey = ""; credentialMessage = "Enregistrement simulé. Aucune clé conservée."; return }
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
        guard !isIsolated else { return }
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

    private var hasStoredCredential: Bool {
        !isIsolated && MistralCredentialStore.hasKeychainCredential
    }

    private var tokenExists: Bool {
        guard !isIsolated else { return false }
        return FileManager.default.fileExists(
            atPath: FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent(".mineru_token").path
        )
    }
}
