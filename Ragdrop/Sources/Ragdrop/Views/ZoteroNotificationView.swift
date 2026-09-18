import SwiftUI

struct ZoteroNotificationView: View {
    @Bindable var monitor: ZoteroMonitorStore
    var body: some View {
        HStack(spacing: 14) {
            Image(systemName: monitor.errorMessage == nil ? "tray.and.arrow.down" : "clock.badge.exclamationmark")
                .font(.title3).foregroundStyle(RagdropTheme.link)
            VStack(alignment: .leading, spacing: 4) {
                if monitor.errorMessage != nil {
                    Text("Zotero monitoring awaiting verification").font(.callout.weight(.semibold))
                    Text("New items are reported only after verification. Check settings.")
                        .font(.caption).foregroundStyle(RagdropTheme.secondary)
                } else {
                    Text("\(RagdropText.pdfCount(monitor.pendingDocuments.count)) from Zotero to review")
                        .font(.callout.weight(.semibold))
                    if let date = monitor.lastRagdocCheck {
                        Text("Duplicates checked at \(date.formatted(date: .omitted, time: .shortened)) · no automatic import")
                            .font(.caption).foregroundStyle(RagdropTheme.secondary)
                    }
                }
            }
            Spacer(minLength: 8)
            if monitor.errorMessage == nil {
                Button("Dismiss") { monitor.dismiss(Set(monitor.pendingDocuments.map(\.attachmentKey))) }
                    .help("Do not report these PDFs again, even if added as another attachment.")
                Button("View new articles") { monitor.showingNewArticles = true }
                    .buttonStyle(RagdropSecondaryButtonStyle())
            }
        }
        .padding(14).ragdropPanel()
    }
}
