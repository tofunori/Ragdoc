#if RAGDROP_SETUP_DEMO
import AppKit
import SwiftUI

@main
struct SetupDemoApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    var body: some Scene {
        WindowGroup("Ragdrop Setup Preview") {
            #if RAGDROP_SETUP_VALIDATION
            VStack(spacing: 0) {
                Text("ISOLATED VALIDATION · temporary library · no real imports").font(.caption).padding(8)
                LocalSetupView(allowServer: false, setup: LocalSetupStore(
                    engine: LocalEngine(support: URL(fileURLWithPath: "/tmp/ragdrop-native-setup/Library/Application Support/Ragdrop"), resources: Bundle.main.resourceURL!),
                    defaults: .standard, allowsPersonalActions: false))
            }.frame(minWidth: 800, minHeight: 780)
            #else
            LocalSetupView(isPreview: true).frame(minWidth: 800, minHeight: 780)
            #endif
        }.defaultSize(width: 860, height: 1000)
    }
}
#endif
