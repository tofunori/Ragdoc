import SwiftUI

struct QueueRowView: View {
    let job: ImportJob
    var onPreview: (() -> Void)?
    var onRetry: (() -> Void)?
    var onShowError: (() -> Void)?
    var onRemove: (() -> Void)?
    var canRemove = true

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: hasRecoverableError ? "exclamationmark.triangle.fill" : job.stage.symbol)
                .font(.title3)
                .foregroundStyle(stageColor)
                .frame(width: 24)
                .symbolEffect(.pulse, isActive: job.stage.isActive)

            VStack(alignment: .leading, spacing: 3) {
                Text(job.displayName)
                    .lineLimit(1)
                Text(job.detail)
                    .font(.caption)
                    .foregroundStyle(hasRecoverableError || job.stage == .failed ? Color.red : Color.secondary)
                    .lineLimit(2)
            }
            Spacer(minLength: 12)
            if job.stage == .awaitingReview || job.stage == .readyForIndexing {
                Button("Aperçu", systemImage: "eye", action: { onPreview?() })
                    .labelStyle(.iconOnly)
                    .help("Prévisualiser le Markdown")
            }
            if job.errorDetails != nil {
                Button("Détails", systemImage: "info.circle", action: { onShowError?() })
                    .labelStyle(.iconOnly)
                    .help("Afficher le détail de l’erreur")
            }
            if job.stage == .failed {
                Button("Relancer", systemImage: "arrow.clockwise", action: { onRetry?() })
                    .labelStyle(.iconOnly)
                    .help("Relancer ce PDF")
            }
            Button("Retirer", systemImage: "trash", role: .destructive, action: { onRemove?() })
                .labelStyle(.iconOnly)
                .help("Retirer de la file sans supprimer l’article de Ragdoc")
                .disabled(!canRemove)
            Text(hasRecoverableError ? "À reprendre" : job.stage.title)
                .font(.caption.weight(.medium))
                .foregroundStyle(stageColor)
        }
        .padding(.vertical, 5)
    }

    private var stageColor: Color {
        if hasRecoverableError {
            return .red
        }
        switch job.stage {
        case .completed: return .green
        case .awaitingReview: return .orange
        case .readyForIndexing: return .green
        case .duplicate: return .orange
        case .rejected: return .secondary
        case .failed: return .red
        case .queued: return .secondary
        default: return .accentColor
        }
    }

    private var hasRecoverableError: Bool {
        job.errorDetails != nil && job.stage == .readyForIndexing
    }
}
