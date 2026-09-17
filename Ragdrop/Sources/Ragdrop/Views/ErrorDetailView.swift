import SwiftUI

struct ErrorDetailView: View {
    let job: ImportJob
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .top) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.title2)
                    .foregroundStyle(RagdropTheme.error)
                VStack(alignment: .leading, spacing: 4) {
                    Text("Échec du traitement")
                        .font(.title2.bold())
                    Text(job.displayName)
                        .foregroundStyle(RagdropTheme.secondary)
                        .lineLimit(2)
                }
                Spacer()
            }

            Text(job.detail)
                .font(.body)

            GroupBox("Détail technique") {
                ScrollView {
                    Text(job.errorDetails ?? job.detail)
                        .font(.system(.caption, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(4)
                }
                .frame(minHeight: 160)
            }

            HStack {
                Spacer()
                Button("Fermer") { dismiss() }
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(RagdropTheme.pagePadding)
        .frame(minWidth: 620, minHeight: 330)
        .ragdropSurface()
    }
}
