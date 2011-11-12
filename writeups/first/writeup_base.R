bw_theme <- function () {
    title_text_h <- theme_text(size = 10, face = "bold")
    title_text_v <- theme_text(size = 10, face = "bold", angle = 90)
    plain_text_h <- theme_text(size = 10)
    plain_text_v <- theme_text(size = 10)

    theme_set(theme_bw(base_size = 10))

    theme_update(
        plot.title = title_text_h,
        legend.title = theme_text(size = 10, face = "bold", hjust = 0),
        axis.title.x = title_text_h,
        axis.title.y = title_text_v,
        axis.text.x = plain_text_h,
        axis.text.y = plain_text_v,
        legend.text = plain_text_h
        )
}

bw_theme()

