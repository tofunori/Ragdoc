import SwiftUI

struct LocalSetupView: View {
    @State private var setup: LocalSetupStore
    @State private var mistralKey = ""
    @State private var voyageKey = ""
    var allowServer = true
    var allowMaintenance = true
    @State private var confirmingRebuild = false
    var onDone: () -> Void = {}
    var onServer: () -> Void = {}

    init(allowServer: Bool = true, allowMaintenance: Bool = true, isPreview: Bool = false, setup: LocalSetupStore? = nil, onDone: @escaping () -> Void = {}, onServer: @escaping () -> Void = {}) {
        self.allowServer = allowServer; self.allowMaintenance = allowMaintenance; self.onDone = onDone; self.onServer = onServer
        _setup = State(initialValue: setup ?? LocalSetupStore(isPreview: isPreview))
    }
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                PageHeading(title: "Your library, on this Mac", subtitle: "Set up once. Review your papers. Ask Claude.")
                if setup.isPreview { InlineNotice(text: "Setup preview · no downloads, keys or library changes.") }
                VStack(alignment: .leading, spacing: 12) {
                    Label("1  Prepare your library", systemImage: setup.isReady ? "checkmark.circle" : "internaldrive")
                        .font(.headline)
                    Text("Ragdrop installs its own Python engine. No terminal, NAS or separate Python installation is needed.")
                        .foregroundStyle(RagdropTheme.secondary)
                    HStack {
                        Text(setup.library.path).font(.caption).textSelection(.enabled).lineLimit(2)
                        Spacer()
                        Button("Choose folder…") { setup.chooseFolder() }.disabled(setup.isPreparing || setup.isReady || !setup.allowsPersonalActions)
                    }
                    Text("Your PDFs stay where they are. Reviewed text, the index and this Mac’s queue are stored here. Existing server libraries are kept separately.")
                        .font(.caption).foregroundStyle(RagdropTheme.secondary)
                    if setup.isPreparing {
                        HStack { ProgressView().controlSize(.small); Text(setup.progress); Spacer(); Button("Cancel") { setup.cancel() } }
                    } else {
                        HStack {
                            Button(setup.isReady ? "Check / repair engine" : "Prepare this Mac") { setup.prepare() }
                                .buttonStyle(RagdropPrimaryButtonStyle()).disabled(setup.isPreview || !allowMaintenance)
                            Text(setup.isReady ? "Engine installed" : "Internet required · first setup may take several minutes")
                                .font(.caption).foregroundStyle(RagdropTheme.secondary)
                        }
                    }
                    if let status = setup.status {
                        if status.needsRepair {
                            InlineNotice(text: "Indexing was interrupted. Rebuild the index before continuing; article counts are not yet verified.", symbol: "exclamationmark.triangle", isError: true)
                            Button("Back up and rebuild index…") { confirmingRebuild = true }
                                .disabled(setup.isPreparing || !allowMaintenance || setup.isPreview || !setup.allowsPersonalActions)
                        } else {
                            Text("\(status.documents) articles · \(status.chunks) passages. Local checks only; no paid search was sent.")
                                .font(.caption).foregroundStyle(RagdropTheme.secondary)
                        }
                    }
                }.padding(20).ragdropPanel()
                VStack(alignment: .leading, spacing: 12) {
                    Label("2  Add your service keys", systemImage: "key").font(.headline)
                    Text("Mistral reads selected PDFs before review. Voyage indexes approved text and processes search queries. These services may charge usage fees; a Claude subscription does not include their API access.")
                        .font(.callout).foregroundStyle(RagdropTheme.secondary)
                    HStack { Text("Mistral OCR"); Spacer(); Text(setup.mistralConfigured ? "Configured" : "Needed for PDF conversion").font(.caption) }
                    SecureField("Mistral API key", text: $mistralKey).textFieldStyle(.roundedBorder)
                    Link("Get a Mistral key", destination: URL(string: "https://console.mistral.ai/api-keys")!)
                    HStack { Text("Voyage AI"); Spacer(); Text(setup.voyageConfigured ? "Configured" : "Needed for indexing and semantic search").font(.caption) }
                    SecureField("Voyage API key", text: $voyageKey).textFieldStyle(.roundedBorder)
                    Link("Get a Voyage AI key", destination: URL(string: "https://dash.voyageai.com/")!)
                    Button("Save keys securely") {
                        setup.saveKeys(mistral: mistralKey, voyage: voyageKey)
                        mistralKey = ""; voyageKey = ""
                    }.disabled(setup.isPreview || !setup.allowsPersonalActions || (mistralKey.isEmpty && voyageKey.isEmpty))
                    if let message = setup.credentialMessage { Text(message).font(.caption).foregroundStyle(RagdropTheme.secondary) }
                }.padding(20).ragdropPanel()
                VStack(alignment: .leading, spacing: 12) {
                    Label("3  Connect Claude Desktop", systemImage: "link").font(.headline)
                    Text("Install the included extension in Claude Desktop. Claude starts the local engine when needed; Ragdrop does not need to stay open. Retrieved passages are shared with Claude when you use the tools.")
                        .foregroundStyle(RagdropTheme.secondary)
                    Button("Install Claude extension…") { setup.connectClaude() }.disabled(!setup.isReady || setup.isPreparing || setup.isPreview || !setup.allowsPersonalActions)
                    Text("Claude will ask you to confirm installation. This extension is for Claude Desktop on this Mac; it does not connect Claude Web or mobile.")
                        .font(.caption).foregroundStyle(RagdropTheme.secondary)
                }.padding(20).ragdropPanel()
                if let error = setup.errorMessage { InlineNotice(text: error, symbol: "exclamationmark.triangle", isError: true).textSelection(.enabled) }
                HStack {
                    if allowServer { Button("Use an existing server") { onServer() }.disabled(setup.isPreparing) }
                    Spacer()
                    Button("Open my library") { onDone() }.buttonStyle(RagdropPrimaryButtonStyle())
                        .disabled(!setup.isReady || setup.isPreparing || setup.isPreview || setup.status?.needsRepair == true)
                }
                Text("You can add keys later in Settings. Importing requires them; preparing an empty library does not upload any documents.")
                    .font(.caption).foregroundStyle(RagdropTheme.secondary)
            }.padding(RagdropTheme.pagePadding).frame(maxWidth: 820)
                .frame(maxWidth: .infinity)
        }.ragdropSurface()
            .task { setup.refreshKeys(); await setup.check() }
            .alert("Rebuild the local index?", isPresented: $confirmingRebuild) {
                Button("Cancel", role: .cancel) {}
                Button("Back up and rebuild") { setup.rebuildIndex() }
            } message: {
                Text("Ragdrop will back up the current library files, then reindex all approved Markdown using Voyage AI. This may incur API charges. Original PDFs are kept. Do not quit until the operation finishes.")
            }
            .interactiveDismissDisabled(setup.isPreparing)
            .onDisappear { setup.cancel() }
    }
}
