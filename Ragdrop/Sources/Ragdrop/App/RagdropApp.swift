import AppKit
import SwiftUI

private enum RagdropSection: Hashable {
    case importer
    case history
    case status
}

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

@main
struct RagdropApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @State private var store = ImportStore()
    @State private var historyStore = HistoryStore()
    @State private var statusStore = RagdocStatusStore()
    @State private var selectedSection: RagdropSection = .importer

    var body: some Scene {
        WindowGroup("Ragdrop", id: "main") {
            Group {
                switch selectedSection {
                case .importer:
                ContentView(store: store)
                case .history:
                    HistoryView(store: historyStore)
                case .status:
                    RagdocStatusView(store: statusStore)
                }
            }
            .frame(minWidth: 680, minHeight: 620)
            .toolbar {
                ToolbarItem(placement: .principal) {
                    Picker("Section", selection: $selectedSection) {
                        Text("Importer").tag(RagdropSection.importer)
                        Text("Historique").tag(RagdropSection.history)
                        Text("État").tag(RagdropSection.status)
                    }
                    .labelsHidden()
                    .pickerStyle(.segmented)
                    .frame(width: 330)
                }
            }
        }
        .defaultSize(width: 760, height: 720)
        .commands {
            CommandMenu("État") {
                Button("Afficher l’état de Ragdoc") {
                    selectedSection = .status
                }
                .keyboardShortcut("e", modifiers: [.command, .shift])
            }
        }

        Settings {
            SettingsView()
        }
    }
}
