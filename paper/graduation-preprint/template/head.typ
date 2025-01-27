#import "./util.typ": spacer

#let mixed(jfont, jweight: "bold", jsize: 1.0em, efont, eweight: "bold", esize: 1.025em, body) = {
  show regex("[\p{Latin}0-9]"): set text(font: efont, weight: eweight, size: esize)
  show regex("[\p{scx:Han}\p{scx:Hira}\p{scx:Kana}]"): set text(font: jfont, weight: jweight, size: jsize)
  body
}


#let head(
  title: (),
  authors: (),
  abstract: "",
  keywords: (),
) = {
  set align(center)

  set text(1.8em)

  if type(title.ja) == "array" {
    grid(
      rows: title.len(),
      row-gutter: 8pt,
      ..title.en.map(t => text(t, font: "Hiragino Mincho ProN", weight: "bold", size: 1.025em)),
    )
  } else {
    grid(
      rows: 1,
      row-gutter: 8pt,
      text(title.en, font: "Hiragino Mincho ProN", weight: "bold", size: 1.025em)
    )
  }

  mixed(
    "Hiragino Kaku Gothic ProN",
    "Hiragino Mincho ProN",
    if type(title.ja) == "array" {
      grid(
        rows: title.len(),
        row-gutter: 8pt,
        ..title.ja,
      )
    } else {
      grid(
        rows: 1,
        row-gutter: 8pt,
        title.ja,
      )
    },
  )

  spacer()

  set text(0.7em)

  grid(
    gutter: 80pt,
    columns: calc.min(authors.len(), 5),
    ..authors.map(author => {
      grid(
        rows: 2,
        row-gutter: 0.8em,
        text(
          font: "Hiragino Mincho ProN",
          weight: "bold",
          size: 1.05em,
          author.en,
        ),
        text(
          font: "Hiragino Kaku Gothic ProN",
          size: 1em,
          author.ja,
        ),
      )
    }),
  )

  spacer()

  pad(
    grid(
      rows: 3,
      row-gutter: 0.8em,
      align(
        center,
        smallcaps(
          text(
            font: "New Computer Modern",
            weight: "bold",
            size: 1em,
          )[Abstract],
        ),
      ),
      align(
        left,
        text(
          font: "Hiragino Mincho ProN",
          size: 0.8em,
          abstract,
        ),
      ),
      align(
        left,
        text(
          font: "Hiragino Mincho ProN",
          size: 0.8em,
        )[#text(font: "New Computer Modern", style: "italic", weight: "bold")[Keywords:] #keywords.join(", ")],
      ),
    ),
    x: 16pt,
  )

  spacer()
}
