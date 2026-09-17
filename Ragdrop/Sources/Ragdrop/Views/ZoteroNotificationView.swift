import SwiftUI

struct ZoteroNotificationView: View {
    @Bindable var monitor: ZoteroMonitorStore
    var body: some View {
        HStack(spacing: 14) {
            Image(systemName: monitor.errorMessage == nil ? "tray.and.arrow.down" : "clock.badge.exclamationmark")
                .font(.title3).foregroundStyle(RagdropTheme.link)
            VStack(alignment: .leading, spacing: 4) {
                if monitor.errorMessage != nil {
                    Text("Surveillance Zotero en attente de vérification").font(.callout.weight(.semibold))
                    Text("Aucune nouveauté n’est annoncée sans contrôle. Consultez les réglages.")
                        .font(.caption).foregroundStyle(RagdropTheme.secondary)
                } else {
                    Text("\(monitor.pendingDocuments.count) nouveau\(monitor.pendingDocuments.count > 1 ? "x" : "") PDF Zotero à examiner")
                        .font(.callout.weight(.semibold))
                    if let date = monitor.lastRagdocCheck {
                        Text("Doublons contrôlés à \(date.formatted(date: .omitted, time: .shortened)) · aucun ajout automatique")
                            .font(.caption).foregroundStyle(RagdropTheme.secondary)
                    }
                }
            }
            Spacer(minLength: 8)
            if monitor.errorMessage == nil {
                Button("Ignorer") { monitor.dismiss(Set(monitor.pendingDocuments.map(\.attachmentKey))) }
                    .help("Ne plus annoncer ces PDF, même s’ils sont ajoutés sous une autre pièce jointe.")
                Button("Voir les nouveaux articles") { monitor.showingNewArticles = true }
                    .buttonStyle(RagdropSecondaryButtonStyle())
            }
        }
        .padding(14).ragdropPanel()
    }
}
