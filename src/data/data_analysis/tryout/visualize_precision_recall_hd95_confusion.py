"""Compatibility wrapper for the renamed HD95-vs-F1 confusion plot script."""

try:
    from .visualize_hd95_f1_confusion import (
        compute_pixel_cls_conf_score,
        is_finite_number,
        load_hd95_f1_confusion_dataframe,
        main,
        plot_hd95_vs_f1_confusion,
        resolve_json_files,
        to_float_or_nan,
    )
except ImportError:
    from visualize_hd95_f1_confusion import (
        compute_pixel_cls_conf_score,
        is_finite_number,
        load_hd95_f1_confusion_dataframe,
        main,
        plot_hd95_vs_f1_confusion,
        resolve_json_files,
        to_float_or_nan,
    )


# Backward-compatible aliases for older imports.
load_plot_dataframe = load_hd95_f1_confusion_dataframe
plot_precision_recall_hd95_confusion = plot_hd95_vs_f1_confusion


if __name__ == "__main__":
    main()
