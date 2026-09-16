import SwiftUI
import UniformTypeIdentifiers

struct DropZoneView: View {
    let isTargeted: Bool
    let chooseFiles: () -> Void
    let chooseZotero: () -> Void

    var body: some View {
        HStack(spacing: 14) {
            Image(systemName: isTargeted ? "arrow.down.doc.fill" : "doc.badge.plus")
                .font(.system(size: 26, weight: .light))
                .foregroundStyle(isTargeted ? Color.accentColor : Color.secondary)
                .symbolEffect(.bounce, value: isTargeted)
                .frame(width: 36)
            VStack(alignment: .leading, spacing: 3) {
                Text(isTargeted ? "Déposez les PDF" : "Glissez vos PDF ici")
                    .font(.headline)
                Text("Détection des doublons, conversion MinerU et aperçu Markdown.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 7) {
                Button("Choisir dans Finder…", systemImage: "folder", action: chooseFiles)
                    .buttonStyle(.bordered)
                Button("Choisir dans Zotero…", systemImage: "books.vertical", action: chooseZotero)
                    .buttonStyle(.bordered)
            }
            .controlSize(.regular)
            .frame(width: 205)
        }
        .frame(maxWidth: .infinity)
        .padding(.horizontal, 16)
        .padding(.vertical, 13)
        .background(.quaternary.opacity(isTargeted ? 0.9 : 0.35), in: .rect(cornerRadius: 14))
        .overlay {
            RoundedRectangle(cornerRadius: 14)
                .strokeBorder(isTargeted ? Color.accentColor : Color.secondary.opacity(0.25),
                              style: StrokeStyle(lineWidth: isTargeted ? 2 : 1, dash: [6]))
        }
        .contentShape(.rect)
    }
}
