import AppKit
import SwiftUI

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        if let iconURL = Bundle.main.url(forResource: "RagdropIcon", withExtension: "icns"),
           let icon = NSImage(contentsOf: iconURL) {
            NSApp.applicationIconImage = icon
        }
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }
}

#if !RAGDROP_REVIEW_DEMO && !RAGDROP_THEME_DEMO && !RAGDROP_SETUP_DEMO
@main
struct RagdropApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @AppStorage("libraryLocation") private var location = LibraryLocation.resolve().rawValue
    @AppStorage("localLibraryPath") private var localLibraryPath = ""
    @State private var importRequest = UUID()
    @State private var selectedSection: WorkspaceSection = .home

    var body: some Scene {
        WindowGroup("Ragdrop", id: "main") {
            WorkspaceSession(section: $selectedSection, importRequest: importRequest)
                .id(location + localLibraryPath)
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 1180, height: 860)
        .commands {
            CommandGroup(replacing: .appSettings) {
                Button("Settings…") { selectedSection = .settings }.keyboardShortcut(",", modifiers: .command)
            }
            CommandGroup(after: .newItem) {
                Button("Import PDFs…") {
                    selectedSection = .home
                    importRequest = UUID()
                }.keyboardShortcut("o", modifiers: .command)
            }
            CommandMenu("Navigation") {
                ForEach(Array(WorkspaceSection.allCases.enumerated()), id: \.element.id) { index, section in
                    Button(section.title) { selectedSection = section }
                        .keyboardShortcut(KeyEquivalent(Character(String(index + 1))), modifiers: .command)
                }
            }
        }
    }
}

private struct WorkspaceSession: View {
    @State private var store = ImportStore()
    @State private var history = HistoryStore()
    @State private var status = RagdocStatusStore()
    @State private var monitor = ZoteroMonitorStore()
    @Binding var section: WorkspaceSection
    let importRequest: UUID
    var body: some View {
        WorkspaceView(store: store, history: history, status: status, monitor: monitor, section: $section)
            .onChange(of: importRequest) { _, _ in store.showingFileImporter = true }
            .onDisappear { monitor.stop() }
    }
}
#endif
