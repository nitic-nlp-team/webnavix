#import "./util.typ": spacer

#let mixed(jfont, jweight: "bold", jsize: 1em, efont, eweight: "bold", esize: 1.025em, body) = {
  show regex("[\p{Latin}0-9]"): set text(font: efont, weight: eweight, size: esize)
  show regex("[\p{scx:Han}\p{scx:Hira}\p{scx:Kana}]"): set text(font: jfont, weight: jweight, size: jsize)
  body
}


#let head(
  title: (),
  authors: (),
  affiliations: (),
  footnotes: (:),
) = {
  set align(center)

  set text(1.8em)

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

  set text(0.7em)

  spacer()

  grid(
    gutter: 1em,
    columns: calc.min(authors.len(), 5),
    ..authors.map(author => {
      let footnote = footnotes.at(str(author.group))

      [#author.ja#super(numbering("*", footnote.index + 1))]
    }),
  )
  grid(
    gutter: 1em,
    columns: calc.min(affiliations.len(), 5),
    ..affiliations.map(author => {
      let footnote = footnotes.at(str(author.group))

      [#super(numbering("*", footnote.index + 1))#author.ja]
    }),
  )

  spacer()
}
