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
            PageHeading(title: "Settings", subtitle: "Conversion, access and destination for your articles.")
            if isIsolated { InlineNotice(text: "Isolated demo · no real key is read or changed.") }
            Form {
            Section("Appearance") {
                Picker("Theme", selection: $appearanceMode) {
                    ForEach(RagdropAppearance.allCases) { appearance in
                        Text(appearance.title).tag(appearance.rawValue)
                    }
                }.pickerStyle(.segmented)
            }

            Section("New Zotero articles") {
                Toggle("Monitor Zotero", isOn: Binding(get: { monitor.isEnabled }, set: { monitor.setEnabled($0) }))
                Text("Report new local PDFs without importing automatically.")
                    .font(.callout).foregroundStyle(RagdropTheme.secondary)
                DisclosureGroup("How it works") {
                    Text("While Ragdrop runs, it checks Zotero’s local API every 5 minutes. Zotero must be open.")
                    Text("When enabled, existing PDFs form a baseline and do not trigger alerts. Use From Zotero to review them. Items without a local PDF will be checked once a PDF is available.")
                }.font(.callout).foregroundStyle(RagdropTheme.secondary)
                if monitor.isEnabled {
                    Text(monitor.statusText).font(.caption).foregroundStyle(RagdropTheme.secondary)
                    if let error = monitor.errorMessage { Text(error).font(.caption).foregroundStyle(RagdropTheme.warning) }
                    HStack {
                        if let date = monitor.lastLocalCheck { Text("Last check: \(date.formatted(date: .omitted, time: .shortened))").font(.caption) }
                        Spacer()
                        Button("Check now") { monitor.checkNow() }.disabled(monitor.isChecking || isIsolated)
                    }
                }
            }

            Section("PDF conversion") {
                Picker("Service", selection: providerBinding) {
                    ForEach(PipelineConfiguration.ConverterKind.allCases) { provider in
                        Text(provider.title).tag(provider)
                    }
                }
                Text("Mistral OCR is the main provider. MinerU is available as a fallback.")
                    .font(.caption)
                    .foregroundStyle(RagdropTheme.secondary)
            }
            Section("Mistral OCR") {
                LabeledContent("API key") {
                    Label(
                        credentialConfigured ? "Configured" : "Missing",
                        systemImage: credentialConfigured ? "checkmark.circle" : "exclamationmark.triangle"
                    )
                    .foregroundStyle(credentialConfigured ? RagdropTheme.success : RagdropTheme.warning)
                }
                SecureField("New Mistral key", text: $newAPIKey)
                    .textFieldStyle(.roundedBorder)
                HStack {
                    Link("Create a key…", destination: URL(string: "https://console.mistral.ai/api-keys")!)
                    Spacer()
                    Button("Delete") { deleteCredential() }
                        .disabled(isIsolated || !hasStoredCredential)
                    Button("Save") { saveCredential() }
                        .buttonStyle(RagdropPrimaryButtonStyle())
                        .disabled(newAPIKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
                if let credentialMessage {
                    Text(credentialMessage).font(.caption).foregroundStyle(RagdropTheme.secondary)
                }
            }
            Section("MinerU — fallback") {
                LabeledContent("Token") {
                    Label(tokenExists ? "Detected" : "Missing", systemImage: tokenExists ? "checkmark.circle" : "exclamationmark.triangle")
                        .foregroundStyle(tokenExists ? RagdropTheme.success : RagdropTheme.warning)
                }
            }
            DisclosureGroup("Advanced settings") {
                TextField("Converter", text: $converterPath)
            }
            Section("Ragdoc on your server") {
                TextField("SSH host", text: $nasHost)
                TextField("Directory", text: $remoteRoot)
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
        guard !isIsolated else { newAPIKey = ""; credentialMessage = "Simulated save. No key stored."; return }
        do {
            try MistralCredentialStore.save(newAPIKey)
            newAPIKey = ""
            credentialConfigured = true
            credentialMessage = "Key saved to macOS Keychain."
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
                ? "An external key is still configured."
                : "Key removed from macOS Keychain."
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
