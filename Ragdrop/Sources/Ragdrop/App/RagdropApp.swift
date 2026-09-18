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

#if !RAGDROP_REVIEW_DEMO && !RAGDROP_THEME_DEMO
@main
struct RagdropApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @State private var store = ImportStore()
    @State private var historyStore = HistoryStore()
    @State private var statusStore = RagdocStatusStore()
    @State private var monitor = ZoteroMonitorStore()
    @State private var selectedSection: WorkspaceSection = .home

    var body: some Scene {
        WindowGroup("Ragdrop", id: "main") {
            WorkspaceView(store: store, history: historyStore, status: statusStore, monitor: monitor, section: $selectedSection)
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 1180, height: 860)
        .commands {
            CommandGroup(after: .newItem) {
                Button("Import PDFs…") {
                    selectedSection = .home
                    store.showingFileImporter = true
                }.keyboardShortcut("o", modifiers: .command)
            }
            CommandMenu("Navigation") {
                ForEach(Array(WorkspaceSection.allCases.enumerated()), id: \.element.id) { index, section in
                    Button(section.title) { selectedSection = section }
                        .keyboardShortcut(KeyEquivalent(Character(String(index + 1))), modifiers: .command)
                }
            }
        }
        Settings { SettingsView(monitor: monitor).ragdropSurface().frame(width: 680, height: 700) }
    }
}
#endif
