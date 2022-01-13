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
cp_file <- file.path("data", "MOA_LSA_zscores.tsv")
cp_df <- readr::read_tsv(cp_file)

cp_df$model <- factor(
    cp_df$model,
    levels = c("Vanilla", "Beta", "MMD", "PCA", "Complete")
)

head(cp_df)

# Compare performance means across architectures
cp_mean_df <- cp_df %>%
    dplyr::group_by(model) %>%
    dplyr::mutate(
        model_mean = round(mean(zscore), 2)
    ) %>%
    dplyr::select(model, model_mean) %>%
    dplyr::distinct()

cp_mean_df

plot_gg <- (
    ggplot(cp_df, aes(x=zscore))
    + geom_histogram(aes(y = ..density..), bins=20)
    + geom_density()
    + geom_text(data=cp_mean_df, aes(x = -2, y = 0.35, label = paste0("Mean: ", model_mean)))
    + geom_vline(data=cp_mean_df, aes(xintercept = model_mean), color = "red", linetype="dashed")
    + facet_wrap("model", nrow = 5)
    + geom_vline(xintercept = 0, linetype = "dashed", color = "blue")
    + theme_bw()
    + plot_theme
    + ylab("Density")
    + xlab("Z-score of L2 distances between real and simulations\nfor predicted polypharmacology state compared to 10 random permutations")
    
)
plot_gg

# Save figure
output_file_base <- file.path("output", "sup_fig_lsa_zscores")
for (file_extension in file_extensions) {
    output_file <- paste0(output_file_base, file_extension)
    ggplot2::ggsave(output_file, plot_gg, dpi = 500, width = 6, height = 8)
}
