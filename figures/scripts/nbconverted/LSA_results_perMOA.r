suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggplot2))

# Set themes
file_extensions <- c(".png", ".pdf")

plot_theme <- theme(
    title = element_text(size = 9),
    axis.title = element_text(size = 9),
    legend.text = element_text(size = 7),
    legend.title = element_text(size = 9),
    legend.key.size = unit(0.5, "cm"),
    strip.text = element_text(size = 10),
    strip.background = element_rect(colour="black", fill="#fdfff4")
)

# Load L2 distances per MOA
cp_file <- file.path("data", "MOA_LSA_metrics.tsv")
cp_df <- readr::read_tsv(cp_file)

cp_df$model <- factor(
    cp_df$model,
    levels = c("Vanilla", "Beta", "MMD", "PCA", "Complete")
)

head(cp_df)

# Compare performance means across architectures
cp_mean_df <- cp_df %>%
    dplyr::group_by(model, metric) %>%
    dplyr::mutate(
        model_mean = round(mean(zscore), 2)
    ) %>%
    dplyr::select(model, metric, model_mean) %>%
    dplyr::distinct()

cp_mean_df

plot_gg <- (
    ggplot(cp_df, aes(x=zscore))
    + geom_histogram(aes(y = ..density..), bins = 30)
    + geom_density()
    + geom_text(
        data=cp_mean_df,
        aes(x = -3, y = 0.5, label = paste0("Mean: ", model_mean))
    )
    + geom_vline(
        data=cp_mean_df,
        aes(xintercept = model_mean),
        color = "red",
        linetype="dashed"
    )
    + geom_vline(
        xintercept = 0,
        linetype = "dashed",
        color = "blue"
    )
    + facet_grid("model~metric")
    + theme_bw()
    + xlim(c(-5, 5))
    + plot_theme
    + geom_rect(
        data = cp_df %>%
            dplyr::filter(model == "MMD"),
        fill = NA,
        alpha = 1,
        color = "red",
        linetype = "solid",
        size = 2,
        xmin = -Inf,
        xmax = Inf,
        ymin = -Inf,
        ymax = Inf
    )
    + ylab("Density")
    + xlab("Z-score of metrics (L2 distance or Pearson correlation) for LSA predictions of\nground truth (per polypharmacology state) compared to 10 randomly shuffled LSA permutations")
    
)
plot_gg

# Save figure
output_file_base <- file.path("output", "sup_fig_lsa_metrics")
for (file_extension in file_extensions) {
    output_file <- paste0(output_file_base, file_extension)
    ggplot2::ggsave(output_file, plot_gg, dpi = 500, width = 8, height = 8)
}
