#import "./head.typ": head
#import "./util.typ": spacer

#let template(
  title: (),
  authors: (),
  affiliations: (),
  body,
) = {
  if type(title.ja) == "array" {
    set document(
      author: authors.map(a => a.ja),
      title: title.ja.join(""),
    )
  } else {
    set document(
      author: authors.map(a => a.ja),
      title: title.ja,
    )
  }

  set page(
    columns: 2,
    binding: right,
    margin: (
      top: 30mm,
      bottom: 25mm,
      x: 20mm,
    ),
    paper: "a4",
  )

  set columns(gutter: 7mm)

  set heading(numbering: "1.1.1")
  show heading: it => {
    text(
      font: "Hiragino Kaku Gothic ProN",
      weight: "bold",
      it,
    )
    pad(spacer(), bottom: -1em)
  }

  set par(
    justify: true,
    spacing: 1em,
    first-line-indent: 1em,
  )

  set text(
    font: "Hiragino Mincho ProN",
    size: 0.8em,
  )

  set list(indent: 1em)
  show list: it => {
    pad(y: 0.1em, it)
  }

  set figure(
    numbering: "1.1.1",
    gap: 10pt,
    supplement: "図",
  )
  show figure: it => {
    it
    spacer()
  }
  show figure.where(kind: table): set figure.caption(position: top)

  let footnotes = (:)
  for author in authors + affiliations {
    let is_exist = footnotes.keys().contains(str(author.group))
    let index = 0

    if not is_exist {
      footnotes.insert(
        str(author.group),
        (
          index: footnotes.len() + 1,
          text: author.en,
        ),
      )
    } else {
      let text = footnotes.at(str(author.group)).text
      footnotes.insert(
        str(author.group),
        (
          index: footnotes.at(str(author.group)).index,
          text: text + ", " + author.en,
        ),
      )
    }
  }

  set footnote.entry(indent: 0em)

  place(
    top + center,
    scope: "parent",
    float: true,
    head(
      title: title,
      authors: authors,
      affiliations: affiliations,
      footnotes: footnotes,
    ),
  )

  set footnote(numbering: "*")
  pad(bottom: -2.6em)[
    #footnote(
      text(font: "Hiragino Mincho ProN", weight: "bold")[#title.en],
      numbering: (..args) => {
        none
      },
    )
    #for f in footnotes {
      footnote(
        str(f.at(1).text),
        numbering: (..args) => {
          // "†"
          none
        },
      )
    }
  ]

  counter(footnote).update(0)
  set footnote(numbering: "1")

  body

  bibliography("../ref.bib", title: "参考文献", style: "ieee")
}

#let i = h(1em)

#let es = h(0.25em, weak: true)

#let tbl(tbl, caption: "") = {
  figure(
    tbl,
    caption: figure.caption(caption, position: top),
    supplement: [表],
    kind: "table",
  )
}

#let img(img, caption: "") = {
  figure(
    img,
    caption: caption,
    supplement: [図],
    kind: "image",
  )
}

