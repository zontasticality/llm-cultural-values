#set document(title: "Cultural Completion Sampling Report")
#set page(margin: 2cm)
#set text(font: "New Computer Modern", size: 10pt)
#set heading(numbering: "1.")
#set par(justify: true)

#align(center)[
  #text(size: 16pt, weight: "bold")[Cultural Completion Sampling Report]
  #v(0.3em)
  #text(size: 11pt)[All Templates, Languages, and Sample Completions]
  #v(0.3em)
  #text(size: 9pt, fill: gray)[Generated from culture.db — v2 classifier, trimmed prompts, lang-matched]
]

#v(1em)

This document shows every prompt template used in the cultural completion experiment, 
its translation into each of 27 languages, and 3 randomly sampled completions per language 
with their v2 classifier labels. Completions are drawn from diverse models where possible. 
Newlines in completions are shown as `\\n`. Completions are truncated at 350 characters.

#pagebreak()
= belief
#text(size: 9pt, fill: gray)[Cultural target: worldview]

== Arabic (ara) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "أؤمن بأن"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=2] \
  #text(fill: rgb("#333"))[ّ الجدية تظهر مدى مهنية و احترافية الشخص.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[العمل التطوعي يسهم في تحويل فكرة التغيير التي تتبناها منظمة أو جمعية إلى واقع على أرض الواقع، إن مساهمة المتطوع في رفع مستوى الوعي بأهمية المتطوعين في المجتمع، إضافة إلى تبادل الخبرات العلمية والإنسانية، وأرى أن العمل التطوعي يحسن من أداء المن]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[ني استحق الحب. قول أن أؤمن بتلك العبارة.\nموضوع]
]

== Bulgarian (bul) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Вярвам, че"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[днес ще бъде приет новият закон за МВР]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_bul] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[ще победим. Познавам отлично Левски. Видяхте как се представиха ЦСКА и Лудогорец. Искам да играем добър футбол]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[emotional_state] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=2] \
  #text(fill: rgb("#333"))[алкохолиците и наркоманите не са на позицията. Колко от нашите "масажисти" не могат да бъдат намерени "с пръчка"? От вида на мола на лицето на д-р Прим]
]

== Czech (ces) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Věřím, že"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=2 SS=4] \
  #text(fill: rgb("#333"))[ve Vánocích se v lidských srdcích najde kousek vřelosti a lásky navíc.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[occupation_achievement] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=2] \
  #text(fill: rgb("#333"))[nejde snad o vyřešení složitějších algebraických rovnic.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_ces] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[jsem pro vás v této rubrice nepsala naposledy.]
]

== Danish (dan) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Jeg tror, at"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_dan] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[jeg har den største kærlighed til blomster, og jeg kan ikke undgå at få et mindre chok, når jeg ser en smuk blomst, jeg ikke har set før, for at se, hvad der kan få mit hjerte til at hoppe.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[titlen var en metafor, som betydde at Løgstrup mente, at han ville fortælle en sand historie - altså ikke en opdigtet historie.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=2] \
  #text(fill: rgb("#333"))[det er meget vigtigt, at man finder en eller anden måde, hvorpå man også kan bringe en vis grad af indhold med ud i skolegangen.]
]

== German (deu) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Ich glaube, dass"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[in diesem Thread eine Diskussion wie beim HSV über die "Macht der Fans" unnötig und fehl am Platz ist.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_deu] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=4 SS=4] \
  #text(fill: rgb("#333"))[es an der Zeit ist, dass wir in der Lage sind, uns selbst und unsere Umwelt besser zu verstehen.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[viele Menschen in der heutigen Zeit das Thema Sterbebegleitung falsch definieren.]
]

== Greek (ell) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Πιστεύω ότι"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[η παραπάνω πρόταση θέλει διατύπωση. Γιατί μπορεί να παρεξηγηθεί.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[τα περισσότερα εστιατόρια που παραγγέλνουν για παράδοση ή παράδοση στο Ηνωμένο Βασίλειο επιθυμούν να το κάνουν με τον πιο αποτελεσματικό τρόπο]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_ell] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[οι περισσότεροι έχετε διαβάσει το άρθρο για τα προβλήματα που δημιουργούν τα στενά παπούτσια. Αυτό δεν είναι ένα από τα καλύτερα άρθρα που έχω διαβάσει για τα προβλήματα που δημιουργούν τα στενά παπούτσια. Παρόλα αυτά]
]

== English (eng) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "I believe that"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[the government needs to invest more in healthcare in order to get rid of diseases, and more time is required to train people to be doctors and nurses in order to avoid misdiagnosed and medical errors in the healthcare industry.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[it is possible to do it, without having to use other data structure (at least, it looks like this to me).]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[occupation_achievement] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=2 SS=2] \
  #text(fill: rgb("#333"))[American became so successful because of hardworking and sacrifice.]
]

== Estonian (est) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Ma usun, et"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[tegelikult on kõigil inimestel, kes on juba ammu töötanud või ka tööd otsivad, olnud selline, et tegelik ja ametlik palk ei ole sama.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[soov ei ole valusaks teha.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=2] \
  #text(fill: rgb("#333"))[ta teeb ise ka ikka midagi\nMa ütleks, et nii see täpsemalt ei käi küll (kuna sa ei pea tööle tulema mitte oma valikuga, vaid see valikud on tehtud) ja sellele vaatamata üritad sa siiski võrgustikes enda kuvandit tõsta.]
]

== Finnish (fin) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Uskon, että"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[ihmiset ovat kuitenkin oikeassa.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[jokaisella meistä on oma ”elämäntarina”, joka kulkee läpi koko elämän.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[emotional_state] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[tämä juttu puhuttelee monia.]
]

== French (fra) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Je crois que"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_fra] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[l’on n’a jamais autant parlé de l’Inde qu’en ce moment.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[nous souhaitons tous un monde où chacun respecterait ses responsabilités et assumerait son rôle dans la société.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[si tu lui laisse à manger, à l'année prochaine tu n'aura plus de grenouille\nen même temps faut bien nourrir quelque chose !]
]

== Hindi (hin) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "मेरा मानना है कि"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[हमें अपने दैनिक जीवन में अपने विचारों को सकारात्मक और रचनात्मक विचारों में बदलने और नकारात्मक विचारों से मुक्त होने की कोशिश करनी चाहिए। हम अक्सर अपने आपको और दूसरों को नकारात्मक और निराशावादी विचारों के बारे में सोचने और बोलने का मौका देते हैं]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[हमें उन लोगों के साथ रहना चाहिए जिनसे हमारा रिश्ता अच्छा हो]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[occupation_achievement] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=2] \
  #text(fill: rgb("#333"))[हम जैसे दुनियाभर में मौजूद अन्य अखबारों के सहयोगी हैं]
]

== Croatian (hrv) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Vjerujem da"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[na vašem području ima dovoljno prostora (ne mislim na zemlju nego na razne objekte) na kojima biste mogli postaviti svoje solarne panele kako bi stvarali vlastitu električnu energiju.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=2] \
  #text(fill: rgb("#333"))[mogu povećati motivaciju na  „šećuca stolica“ gdje učenici mogu vidjeti napredak.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[je ova tema zanimljiva mnogim mladim (možda i starijim) ženama.]
]

== Hungarian (hun) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Hiszem, hogy"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[ezt sokan gondolhatják.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[occupation_achievement] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=2] \
  #text(fill: rgb("#333"))[a legjobb megoldást a legjobb szakemberek tudják előállítani.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_hun] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[mindenki képes a saját útját járni, és ebben a folyamatban megtalálni a maga kapaszkodóit és felvenni a kesztyűt az őt akadályozó tényezőkkel szemben, legyen az akár csak egy aprócska szikrányi figyelmetlenség.]
]

== Italian (ita) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Credo che"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_ita] #h(0.5em) #text(weight: "bold")[occupation_achievement] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[il lavoro di un grafico abbia a che fare con l'arte e la poesia, con il piacere di creare e di inventare.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[un sogno così possa aiutare i bambini di 4, 5 e 6 anni a spiegare non solo chi è un animale domestico, ma anche a comprendere meglio le sue caratteristiche e il motivo per cui è necessario farlo lavorare nella vita di tutti i giorni.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[il libro che non abbia ancora scritto ma che non vedo l’ora di scrivere sia un libro su tutto ciò che c’è dietro le quinte.]
]

== Lithuanian (lit) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Tikiu, kad"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[atvirai.\nDauguma tokių "paskolų" įvykdo.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[lankiau būdvardį, kaip jis įtrauktas į pasakojimą.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[Jūs jaučiatės gerai, jei turite gerus draugus, kuriuos gali priimti kaip šeimos narius.]
]

== Latvian (lvs) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Es ticu, ka"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[pārnadžu klase tiek dalīta divās kārtās: liellopiem un nagaiņiem.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_lvs] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[2020. gads būs lielu pārmaiņu gads.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=5] \
  #text(fill: rgb("#333"))[tēlotāja māksla ir tā, kurā var izpausties mani vislielākie sasniegumi, uz ko spēj mani ļaut spēki!]
]

== Dutch (nld) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Ik geloof dat"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=4 SS=4] \
  #text(fill: rgb("#333"))[de mensheid meer te maken heeft met een gebrek aan bewustzijn dan met een gebrek aan middelen.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=3] \
  #text(fill: rgb("#333"))[ik me er maar voor ga uitspreken, aangezien het al in meerdere topic kwam.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=5 TS=1 SS=3] \
  #text(fill: rgb("#333"))[familie en opvoeding altijd boven alles hebben gestaan.]
]

== Polish (pol) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Wierzę, że"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[każdy ma w sobie wielkie pokłady kreatywności.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[prawdziwi przyjaciele nie odchodzą w niepamięć, dlatego będę do Ciebie szukać, nawet jeśli Ty nie będziesz szukał mnie.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_pol] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[w każdym z nas jest coś z dziecka, bo bez tego dziecko nie istnieje.]
]

== Portuguese (por) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Eu acredito que"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_por] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[todos os sonhos tem um significado.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[você é o que faz, acredito que devemos ser humanos com os próximos, independentemente das condições sociais e da condição financeira.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[occupation_achievement] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[o Brasil tem grande potencial para se tornar um país rico, forte e desenvolvido.]
]

== Romanian (ron) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Cred că"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[voi declara - vraKnow.com\nExpresiile mai tehnice ale competenței instrumentale sunt mai potrivite în exprimarea opiniilor]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_ron] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=2 SS=3] \
  #text(fill: rgb("#333"))[a devenit un obicei ca, odată cu fiecare concert de colinde, să vină și articolul ăsta.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[physical_attribute] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=2] \
  #text(fill: rgb("#333"))[am fost una dintre cele mai mari fete din clasa a cincea.]
]

== Slovak (slk) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Verím, že"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_slk] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[každý z vás sa raz bude musieť zamyslieť nad tým, čo je pre vás naozaj v živote naozaj dôležité a čo nie.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[sa nachádzate na odľahlej planéte, kde chcem, aby ste sa cítili ako doma.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=2 SS=3] \
  #text(fill: rgb("#333"))[každý sa aspoň raz vo svojom živote zúčastnil na krste dieťaťa.]
]

== Slovenian (slv) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Verjamem, da"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[nisem prva, ki se je vprašala, od kod sploh prihaja krompir?]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[se je vsak od nas kdaj našel v situaciji, ko ne ve, kaj bi spekel za malico in tudi kdaj v zadnjem hipu, ko nam gostje pridejo na obisk.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_slv] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=1] \
  #text(fill: rgb("#333"))[se na področju nege kože in kozmetike dogajajo spremembe, ki se jih veselimo vse pogosteje, saj se bo to področje v prihodnosti izkazalo kot izjemno donosno.]
]

== Spanish (spa) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Creo que"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_spa] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=1 TS=4 SS=3] \
  #text(fill: rgb("#333"))[nunca he estado más en desacuerdo con la prensa y con la mayoría de los españoles.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[emotional_state] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[muchos de vosotros os acordaréis del episodio de "V de Vendetta" cuando se incendió el primer ministro y la gente se levantó en armas contra el tirano y su gobierno.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[tu hijo tiene autismo]
]

== Swedish (swe) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Jag tror att"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[physical_attribute] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=2] \
  #text(fill: rgb("#333"))[min historia står med i något häfte på biblioteket, sa den 92-årige, svaga och blyga Gösta.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[det är i de allra flesta av oss en strävan att utvecklas, att vilja utvecklas.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_swe] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[jag har börjat med något nytt, något jag inte gjort förut men något jag vill göra.]
]

== Turkish (tur) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "İnanıyorum ki"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=5 TS=3 SS=4] \
  #text(fill: rgb("#333"))[; bizler, yepyeni bir dünyaya, şafak vakti doğan bir güneşin ışığında gözlerimizi açmış gibiyiz.\nİnsanlar olarak, dünyanın hangi bölgesine ait olursak olalım, bütün insanlar kardeşimizdir.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[gün geçtikçe artan ihtiyaçlardan dolayı, öğrencilerin bütçesi, fiziksel durumu ya da ailelerinin ulaşabildikleri önlemleri doğrultusunda, öğrenciler zeka kartları kullanmak istiyor ve bu kartların uygulama çeşidi ve uygulama alanının geniş olmasından dolayı başlıca ihtiyaç haline gelmektedir.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=4 SS=4] \
  #text(fill: rgb("#333"))[, bu yeni nesil, bir çok yeniliği daha gerçekleştirecek.]
]


#pagebreak()
= childrearing
#text(size: 9pt, fill: gray)[Cultural target: socialization]

== Arabic (ara) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "يجب على الأطفال أن يتعلموا"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[كيفية التصرف خلال الطوارئ أو الكوارث قبل حدوثها، ويجب أن يبقوا آمنين إذا كانت الأمور سيئة، وإذا كنت تقوم بعملية إخلاء في المنزل، فتعرف على الأماكن التي يمكنك الهروب إليها عندما يبدأ الحريق أو أن تتعلم مهارات البقاء على قيد الحياة عندما تصبح]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[طريقة التفكير بدون الأستناد علي المظهر الخارجي للأشياء من حين لآخر كقاعدة أو المظهر الخارجي وحده.\nيساعدكِ على تعليم أطفالك العديد من الأشياء الحياتية، بالإضافة إلى منحهم دفعة نفسية تحملهم على مواصلة العمل من أجل تحقيق أحلامهم، علمِ أ]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[مبادئ وقواعد وأمور الإتيكيت وذلك لكي تنضبط حياتهم الاجتماعية في المجتمع الذي يعيشون فيه. كما أن الإتيكيت في المنزل مهم جداً لكي يتحول المنزل إلى مكان آمن ومريح يحب كل فرد من أفراد الأسرة أن يعيش فيه. كما ويجب أن يتعلم الأطفال آداب الإتيكيت منذ الص]
]

== Bulgarian (bul) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Децата трябва да се научат да"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[боравят с ножа. Разбира се]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_bul] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[обичат тялото си и да правят здравословен избор на храна]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[occupation_achievement] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=1] \
  #text(fill: rgb("#333"))[ценят работата]
]

== Czech (ces) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Děti by se měly naučit"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[zacházet s elektronickými prostředky odpovědně.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[, že v životě není všechno jednoduché.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[že, hubení hmyzu by mělo probíhat ve vhodnou denní a roční dobu.]
]

== Danish (dan) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Børn bør lære at"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=2] \
  #text(fill: rgb("#333"))[tage gode billeder.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_dan] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[spise sundt.\nDet er vigtigt, at børn lærer at spise sundt og varieret.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[beskrive andre mennesker\nBørn bør lære at beskrive andre mennesker.]
]

== German (deu) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Kinder sollten lernen,"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=3] \
  #text(fill: rgb("#333"))[Verantwortung für ihr Handeln zu übernehmen.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[Konflikte in der Pause zu lösen\nBerlin (dpa/tmn) - Sie schubsen oder schlagen oder beschimpfen einen anderen.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_deu] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[wie man sich bei einer Schlange verhält.]
]

== Greek (ell) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Τα παιδιά πρέπει να μάθουν να"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[χρησιμοποιούν τα διάφορα αντικείμενα που έχουν ανάγκη (ντουλάπα]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_ell] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[μοιράζονται τα πράγματα τους]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[αντιδρούν σε μια έκτακτη ανάγκη. Οι καλύτερες δεξιότητες πρώτων βοηθειών μπορεί να σώσουν ζωές. Το Πρόγραμμα πρώτων βοηθειών του Ελληνικού Ερυθρού Σταυρού δίνει στα παιδιά τη γνώση και την αυτοπεποίθηση που χρειάζονται για να γνωρίζουν]
]

== English (eng) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Children should learn to"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_eng] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=1] \
  #text(fill: rgb("#333"))[read, write, and do arithmetic in Grade 1, but what about Grade 2?]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[listen in a respectful manner and to respond with respect, as well as to use respectful ways of speaking to others.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=1 SS=3] \
  #text(fill: rgb("#333"))[respect their mother and father at an early age.]
]

== Estonian (est) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Lapsed peaksid õppima"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[juba esimeses kodutöös tundma peamistest ja kuumastast paber, mis on kõige sagedamini kasutada.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_est] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[juba maast madalast, et tulevikus nii endale, kui ka teistele palju head teha.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[matemaatikat lihtsast keeruliseks.]
]

== Finnish (fin) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Lasten pitäisi oppia"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[tuntemaan erilaisia eläimiä ja niiden elinympäristöä, mikä on tärkeää ympäristön ja luonnon suojelun kannalta.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[tuntemaan ruokavalikoimansa ja ymmärtää, että maapallolla on herkät luonnonsuhteet.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_fin] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[syömään puuroa\nJos sinulla on lapsia, tiedät, että he eivät ole täysin oikeaoppisia.]
]

== French (fra) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Les enfants devraient apprendre à"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[devenir de bons citoyens du monde\nLe dépôt de gerbe\nSuite de la cérémonie au monument aux morts.\n31 janvier 2018 à 9h03 par Corine Mandin\nAlors que les cérémonies de la Saint-Nicolas battaient leur plein dans la matinée du 6 décembre dernier]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[conduire à l’école\n\nIl semblerait que les cours de conduite à l’école soient de plus en plus populaires.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[cuisiner jeunes avec leur famille.]
]

== Hindi (hin) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "बच्चों को सीखना चाहिए"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=2] \
  #text(fill: rgb("#333"))[ताकि वे अपने परिवेश के बारे में और अधिक जागरूक हो सकें और यह एक सामान्य प्रश्न हो सकता है कि क्या सीखना है।\nEnglish to Hindi Dictionary app is very helpful to students to understand the meaning of any English words\nइस ऐप में हर हिंदी शब्द का उपयोग बहुत आसान भाषा में किया जाता है]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[कि एक अच्छा जीवन जीते हैं\nबच्चों को सीखना चाहिए कि एक अच्छा जीवन जीने के लिए। खेल के द्वारा ज्ञान प्राप्त करने के लिए एक अद्भुत तरीका है। अपने बच्चे के अंदर छिपा हुआ साहसी बाहर लाने और उसे विभिन्न रंग]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=2 SS=3] \
  #text(fill: rgb("#333"))[कि उन्हें कहां रखना चाहिए। एक नकारात्मक अनुशासनात्मक उपाय (जो 2 वर्ष से कम उम्र के बच्चों पर लागू नहीं होता है) उन्हें उनके बिस्तर पर भेजने के लिए (और वहां रहने के लिए]
]

== Croatian (hrv) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Djeca bi trebala naučiti"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=2] \
  #text(fill: rgb("#333"))[pisati abecedu prije 1. razreda škole – Jabuka.tv\nVješti pisac pokazuje sposobnost praktičnog i kreativnog korištenja raznih sposobnosti i vještina.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[što znači imati čistu savjest, te što se događa kada netko to nema, smatraju hrvatski stručnjaci za odgoj.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[osnovna matematička računala iznad 5 godina i do 10 godina starosti.]
]

== Hungarian (hun) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "A gyerekeknek meg kell tanulniuk"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[felismerni és kódolni a tudásukat a környezetünkről, különös tekintettel a veszélyhelyzetek elkerülésére (mozgás, közlekedés, építészet).]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[, hogy a kutyák olyan élőlények, akiket szeretni és vigyázni kell.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[sok fontos dolgot az életben.]
]

== Italian (ita) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "I bambini dovrebbero imparare a"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_ita] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[dormire? Secondo uno studio dell’Università di Newcastle, pubblicato sul British Medical Journal, l’ideale sarebbe 10 ore di sonno ogni notte.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=5 SS=1] \
  #text(fill: rgb("#333"))[usare il computer il più precocemente possibile.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[pensare a se stessi come “eroi” in grado di salvare se stessi e gli altri grazie alla loro forza interiore.]
]

== Lithuanian (lit) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Vaikai turėtų išmokti"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=2] \
  #text(fill: rgb("#333"))[valgyti šaukšteliu (šakute) bent 1,5 metų, kai jie išaugo iš šepetėlio.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[kritiškai mąstyti - DELFI Gyvenimas\nApie problemų sprendimo ir kritinio mąstymo ugdymą daug kalbama vadovėliuose bei mokymo strategijose. Šios ugdymo kryptys pradėtos taikyti prieš gerą dešimtmetį]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[spiritual_religious] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=1 SS=3] \
  #text(fill: rgb("#333"))[iš anksto, o ne paskui\n\nKai Jūsų vaikas bus pakankamai subrendęs, norėdamas priimti atsakingą sprendimą dėl jo širdies, jis turės būti tvirtai įsišaknijęs į savo tikėjimo pagrindus.]
]

== Latvian (lvs) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Bērniem vajadzētu iemācīties"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_lvs] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[uzklausīt citus cilvēkus un sadarboties ar tiem.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[, kā rūpēties par sevi un attīstīt veselīgus ieradumus.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[physical_attribute] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=2] \
  #text(fill: rgb("#333"))[peldēt agri un viegli.]
]

== Dutch (nld) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Kinderen moeten leren om"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[emotional_state] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[gaan met verlies en rouw.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[gaan met geld.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_nld] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[hun eigen beslissingen te nemen, maar soms moet je ze helpen om die beslissingen te nemen.]
]

== Polish (pol) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Dzieci powinny uczyć się"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[angielskiego. – Adajslo.pl\nMogłoby się wydawać, że Polacy to bardzo niewielka grupa narodowa mieszkająca w Azji.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[od najmłodszych lat życia, bo w późniejszym okresie trudno je będzie przekonać do aktywności i zainteresować wyobraźnią.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=2] \
  #text(fill: rgb("#333"))[języka, żeby dobrze się w nim porozumiewać.]
]

== Portuguese (por) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "As crianças devem aprender a"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[resolver conflitos sem violência? - BLOG | Casa Educar\njul 23, 2018 | casa Educar, Comportamento, Mundo infantil | 1 Comentário\n“A criatividade das crianças é tão colorida que não pode caber dentro dos nossos rótulos e regras”, Marisa.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[dar valor a cada coisa.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[importância de ajudar os outros, da maneira como é mais fácil para cada uma delas.]
]

== Romanian (ron) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Copiii ar trebui să învețe să"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=1 SS=3] \
  #text(fill: rgb("#333"))[asculte (verbului) și să adore (adjectivul), dacă vor să crească bine.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[scrie doar cu ajutorul unei creioane.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[facă un lucru până la capăt, iar responsabilitatea lor să nu se sfârșească atunci când părăsesc școala, ci să acționeze și acasă, în cartier, în oraș, așa cum își doresc ei să se comporte și în școală.]
]

== Slovak (slk) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Deti by sa mali naučiť"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[emotional_state] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[, čo je zlá nálada, a to predovšetkým ako s ňou pracovať.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=4 SS=4] \
  #text(fill: rgb("#333"))[, ako definovať svoj názor a názor oponenta.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[pravidlá v spoločnosti.]
]

== Slovenian (slv) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Otroci bi se morali naučiti"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[occupation_achievement] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=2] \
  #text(fill: rgb("#333"))[samostojno opraviti svoje naloge in pri tem uporabljati ustrezne metode dela.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[razmišljati in odločati, kako naj razumejo vsebine, ki so jim v okviru vzgojnega programa predstavljene.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_slv] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[razločevati med igro in resničnim življenjem, med resnično in domišljijsko igro, med resničnim in neresničnim doživljanjem, med pravim in namišljenim življenjem.]
]

== Spanish (spa) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Los niños deben aprender a"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[emotional_state] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=5 SS=4] \
  #text(fill: rgb("#333"))[no sentirse incómodos al expresar sus emociones, ni avergonzarse por sentir dolor.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[emotional_state] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=2] \
  #text(fill: rgb("#333"))[perder en el deporte desde pequeños\nLa pérdida es uno de los principales miedos que tienen todos los seres humanos, y en especial los niños.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[reconocer las \<b\>frutas\</b\>, \<b\>legumbres, verduras y hortalizas\</b\> de una forma \<b\>sencilla y divertida\</b\>.]
]

== Swedish (swe) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Barn bör lära sig att"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_swe] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[respektera djur. I sin nuvarande form har den en stor påverkan på djurs liv.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[vara snälla mot djur, visa hänsyn och inte hålla för mycket på med djur, det är en helt annan sak.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[värdera sig själva och känna stolthet över sitt ursprung.]
]

== Turkish (tur) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Çocuklar öğrenmeli"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=2 SS=3] \
  #text(fill: rgb("#333"))[. Büyüklere kalmaz bu işler.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[öğrenmeye devam etmelidir!]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[ve eğlenmelidir. Bu, evde çocuklarla öğretmeyi eğlenceli hale getiren çocuk matematik oyunlarına sahip olmanın nedenidir.]
]


#pagebreak()
= decision
#text(size: 9pt, fill: gray)[Cultural target: agency]

== Arabic (ara) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "عندما أواجه قراراً صعباً،"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[أو يتوجب علي اختيار واحد من بين عدد كبير من الخيارات المتاحة لي، أخبر نفسي أن علي الاختيار من أجل البناء.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[آخذ وقتي في التفكير قبل الإقدام على هذه الخطوة. إذا كان القرار لا يسبب ضرراً جسيماً، أترك الأمر على طبيعته ولا أحاول أن أجعله أفضل، ولكنه ليس دائماً خياراً مناسباً.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[فإنني أبقى هادئا وأفكر ملياً في القضية واتأمل قليلاً\nكنت أخطط لإنشاء أول مشروعي التجاري خلال العام الماضي . و بالفعل بحثت عن القطاع الأنسب لبدء عملي التجاري و وجدت ذلك القطاع المناسب ، كما ساعدني والدي في الحصول على ما أحت]
]

== Bulgarian (bul) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Когато трябва да взема трудно решение,"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_bul] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[често се консултирам с моя лекар. Има специални въпроси]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[това може да ми е от полза"]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=5 SS=3] \
  #text(fill: rgb("#333"))[се допитвам до специалисти и анализирам внимателно фактите]
]

== Czech (ces) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Když musím udělat těžké rozhodnutí,"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[emotional_state] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[velmi mě to zmate.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[které má fatální následky pro nějaký život nebo vztah a nebo mám pochybnosti nad morálností svých činů, ptám se sám sebe a říkám:\n„Co by udělal Jára Cimrman?“\nA když dostanu odpověď, je moje rozhodnutí snadné.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[obracím se na přírodu, která mě usměrňuje.]
]

== Danish (dan) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Når jeg skal træffe en svær beslutning,"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[har jeg brug for at (tænke over/overveje) \_\_\_\_\_\_\_\_\_\nHvad er det sidste, du har taget/modtaget en beslutning om?]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[plejer jeg at spørge mig selv: ”Hvad vil jeg fortælle mine børn om det?”\n\nLæs det igen!]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_dan] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=2] \
  #text(fill: rgb("#333"))[kan jeg godt finde på at søge mod den nemme løsning.]
]

== German (deu) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Wenn ich eine schwierige Entscheidung treffen muss,"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[dann frage ich mich oft, was ich an der Stelle von demjenigen machen würde, der am meisten davon betroffen ist.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[höre ich mich ganz genau ab, wo meine Gedanken hingehen und wie sich das anfühlt.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_deu] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=3] \
  #text(fill: rgb("#333"))[dann gehe ich sehr viel durch, bevor ich eine Lösung finde.]
]

== Greek (ell) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Όταν πρέπει να πάρω μια δύσκολη απόφαση,"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[συνηθίζω να περάσω μια βόλτα από τη σχολή. Εκεί]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=3] \
  #text(fill: rgb("#333"))[συνήθως εμπιστεύομαι την πρώτη μου αντίδραση. \_\_\_\_\_\_\_\_\_\_\_\nΤα κείμενα του Ασκληπιού\nΠρώτα πρώτα]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_ell] #h(0.5em) #text(weight: "bold")[emotional_state] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=2] \
  #text(fill: rgb("#333"))[είμαι ανήσυχος]
]

== English (eng) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "When making a difficult decision, I"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=4 SS=3] \
  #text(fill: rgb("#333"))[always have a general idea of how I want to behave, how I'm going to justify my decision and what reasons I'll give, whether or not those reasons are true.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_eng] #h(0.5em) #text(weight: "bold")[spiritual_religious] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=1 SS=3] \
  #text(fill: rgb("#333"))[often turn to my trusty friends, the scriptures.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=3] \
  #text(fill: rgb("#333"))[am careful not to rush into a decision.]
]

== Estonian (est) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Kui pean tegema raske otsuse,"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[loen situatsiooni ise endale hiljem ette: 'Olukord oli selline...' See aitab palju, sest nii on mul tunne, nagu loeksin olukorda hoopis kõrvalseisjana või näitlejaskuna.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_est] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[kas võtta ette 13 kuud või 2020.aasta, siis mõtlen esmalt sellele, kuidas ma seda teen ja mis minust üldse saab.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[käin sageli ringi, mõtlen palju, kasin oma aeglase otsustamisega kõigi närvid kulutustele ja viimase hetkeni jään ootama midagi imesünget, mis võib minu mõttekaid põhimõtteid ja mõnikord isegi väärtusi rikkuda.]
]

== Finnish (fin) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Kun teen vaikean päätöksen,"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[emotional_state] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[mietin, jos päädyin oikeaan ratkaisuun...]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[pyydän apua ystäviltäni.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_fin] #h(0.5em) #text(weight: "bold")[emotional_state] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=2] \
  #text(fill: rgb("#333"))[joudun myöntämään sen itselleni: tässä käy huonosti.]
]

== French (fra) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Quand je dois prendre une décision difficile,"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_fra] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[j'écoute ma petite voix\nUne autre idée reçue est que nous n'écoutons pas notre petite voix…\nNous avons tous appris à la maîtriser]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[emotional_state] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[un choix qui me fait vraiment mal au ventre, je fais comme un jeu de société.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[spiritual_religious] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=1 SS=4] \
  #text(fill: rgb("#333"))[que je cherche un sens à ma vie, que je m’interroge sur mon orientation… Je consulte les oracles.]
]

== Hindi (hin) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "जब मुझे कोई मुश्किल फ़ैसला लेना होता है, तो मैं"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[क्या करता हूं?]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[spiritual_religious] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=1 SS=3] \
  #text(fill: rgb("#333"))[ऐसे लोगों के विचार पूछने के लिए समय निकालता हूँ जो मेरे विचारों का सम्मान करते हैं और जिनके विचारों से मैं सहमत हूँ.\nपरमेश्वर की सेवा करने में अपनी माँ का साथ देना]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[अपने साथ बैठे लोगों से पूछता हूं कि मुझे क्या करना चाहिए। किसी को अपने मन में क्या चल रहा है]
]

== Croatian (hrv) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Kada moram donijeti tešku odluku,"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[imam savjet s kojega se rijetko skrenem: da li me ova odluka čini boljom osobom?]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[tada moram dobro razmotriti.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[obično pokušavam sve mogućnosti obuzdati i vizualizirati.]
]

== Hungarian (hun) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Amikor nehéz döntést kell hoznom,"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_hun] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[akkor mindig felmerül bennem a kérdés, hogy ha úgyis választanom kell, akkor miért ne tenném azt a legszebb módon?]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[emotional_state] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[vagy valamilyen új helyzet előtt állok]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[mindig a szívemre hallgatok.]
]

== Italian (ita) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Quando devo prendere una decisione difficile,"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[per esempio sulle vacanze, mi prendo un po’ di tempo per pensare.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=2] \
  #text(fill: rgb("#333"))[come scegliere una destinazione per una vacanza]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_ita] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=3] \
  #text(fill: rgb("#333"))[la prima cosa che faccio è quella di analizzare a fondo la situazione.]
]

== Lithuanian (lit) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Kai reikia priimti sunkų sprendimą,"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[emotional_state] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[vieniems jau po n metų atrodo, kad „kadangi jis buvo priimtas prieš daugelį metų“, to „reikia“, o kitiems sunkus sprendimas išlieka „sunkiu“ ir priimti ką nors panašaus ir toliau yra nepaprastai sunku.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_lit] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[daugelis žmonių pirmiausia svarsto kelis variantus.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[mums tenka pasirinkti iš dviejų skirtingų variantų, bet dažnai pamirštame, kad iškyla ir trečioji pasirinkimo alternatyva - tiesiog nedaryti nieko, žengti nė vieno žingsnio.]
]

== Latvian (lvs) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Kad man jāpieņem grūts lēmums,"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_lvs] #h(0.5em) #text(weight: "bold")[emotional_state] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=2] \
  #text(fill: rgb("#333"))[es vienmēr domāju: "vai ir vērts turpināt tūlīt," "vai es tiešām varu kaut ko darīt, lai pārdomātu?" Vai tas ir tā vērts?]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=2] \
  #text(fill: rgb("#333"))[es cenšos pats sevi apmānīt - Noderēs.lv\nKad man jāpieņem grūts lēmums, es cenšos pats sevi apmānīt\nĪstenība, izrādās, ir tendēta uz maldināšanu.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[es atgriezāšos pie šīs grāmatas. Īsās nodaļās un lakoniskā valodā tiek aplūkots no vienas puses klasisks analītisks pieeju problēmas risināšanai un pieņemot lēmumus]
]

== Dutch (nld) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Als ik een moeilijke beslissing moet nemen,"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[dan bedenk ik me vaak: “Ik kan dit zo niet zelf beslissen.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[doe ik altijd een stap terug en bedenk de mogelijkheden.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_nld] #h(0.5em) #text(weight: "bold")[emotional_state] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[zou ik die nooit meer kunnen uitstellen\nals ik wist dat het de laatste zou zijn.]
]

== Polish (pol) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Podejmując trudną decyzję,"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[emotional_state] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=2] \
  #text(fill: rgb("#333"))[by po raz kolejny wyciągnąć ze szuflady te kilka moich wspomnień, które zawsze budziły we mnie niezdrową ciekawość, zdawałem sobie sprawę z tego, że w moim życiu nic się nie zmieniło.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[czy warto odkurzać kanapy i inne meble tapicerowane, najpierw warto zapoznać się z wieloma faktami, które jasno określają, że taka czynność jest bardzo istotna.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=2] \
  #text(fill: rgb("#333"))[jaką jest stylizacja weselna, powinniśmy dokładnie rozważyć każdy wybrany element.]
]

== Portuguese (por) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Quando preciso tomar uma decisão difícil,"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_por] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[sempre utilizo esta ferramenta.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[invariavelmente escolho a opção menos confortável.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[me virei para um método um tanto incomum - clichês.]
]

== Romanian (ron) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Când trebuie să iau o decizie dificilă,"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_ron] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[mă pun în pat.\ncitat din Gene Roddenberry\nAdăugat de Cornelia Georgescu\nComentează! | Votează! | Copiază!]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[spiritual_religious] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=1 SS=3] \
  #text(fill: rgb("#333"))[apelez la Tarot pentru a-mi da câteva indicii.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[aplic tehnica deciziilor școlii.\nEchipa care adună la sfârșit cele mai multe jetoane câștigă.\nDupă ce am condus o perioadă de timp în jurul orașului, am analizat o hartă, iar când am întâmpinat o intersecție am oprit (în condiții de siguranță, desigur).]
]

== Slovak (slk) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Keď musím urobiť ťažké rozhodnutie,"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=2] \
  #text(fill: rgb("#333"))[opýtam sa svojho zrkadla, čo si o tom myslí.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[často si v hlave premietam zoznam ľudí, ktorých milujem.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[spiritual_religious] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=1 SS=3] \
  #text(fill: rgb("#333"))[najprv sa pomodlím a potom použijem hlavu.]
]

== Slovenian (slv) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Ko moram sprejeti težko odločitev,"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[se z vsemi naokoli posvetujem, ker se bojim, da bom na koncu preveč zaželela tistega, česar nisem izbrala.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=1 TS=4 SS=4] \
  #text(fill: rgb("#333"))[nikoli ne berem knjig o reševanju problemov. (Ker mi ni treba dokazati, da bi morali skrbeti za druge, sem morda tako dober v izbiranju naših najboljših možnosti).]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_slv] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[da nekaj izpeljem, se velikokrat vprašam kaj v resnici želim.]
]

== Spanish (spa) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Cuando tengo que tomar una decisión difícil,"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[por ejemplo, la decisión de una relación o de un nuevo empleo, hago una lista y en un lado de la lista, escribo las razones por las que quiero que esto suceda, y en el otro lado de la lista, escribo por qué no debería pasar.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_spa] #h(0.5em) #text(weight: "bold")[emotional_state] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=2] \
  #text(fill: rgb("#333"))[pienso en el pasado, en las oportunidades perdidas, en los errores y en las personas que podrían haberlo hecho mejor.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[siempre le pido consejos a mi esposa.]
]

== Swedish (swe) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "När jag måste fatta ett svårt beslut,"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=2] \
  #text(fill: rgb("#333"))[brukar jag först lägga upp det på listor, så att jag får ett överblickbart underlag.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[när något upprör mig till tårar eller får mig att skrika av ilska har jag en vän till min hjälp.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_swe] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[då ringer jag en vän.]
]

== Turkish (tur) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Zor bir karar vermem gerektiğinde,"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[diyelim ki ailemin nereye tatile gideceğine]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[emotional_state] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[içime dönüyor ve kalbime bakıyorum.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[önce kendi kendime sorarım “En önemlisi hangisi?” Eğer cevabımı bulabilmişsem hemen cevap verebilirim.]
]


#pagebreak()
= family
#text(size: 9pt, fill: gray)[Cultural target: family]

== Arabic (ara) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "في عائلتي، أهم شيء هو"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[أن يكون لدينا إجابة لسؤال واحد فقط]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=5 TS=3 SS=4] \
  #text(fill: rgb("#333"))[أن الجميع يساعدون بعضهم البعض.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=5 SS=4] \
  #text(fill: rgb("#333"))[المكانة، وليس الجنس. لا شيء يمكن أن يغير خيارات النساء في المناصب القيادية. لا ينبغي أن تكون هناك حاجة إلى المرأة لاعتلاء العرش. - فان ثاي.]
]

== Bulgarian (bul) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "В моето семейство най-важното е"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[emotional_state] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[мирът. И не е изненадващо]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_bul] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[да бъдем здрави и да се обичаме. Но ние сме малки хора]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[да няма лъжа]
]

== Czech (ces) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "V naší rodině je nejdůležitější"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[... - PDF\nV naší rodině je nejdůležitější...]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[vnouče. To se musí udělat všem vnučkám, ale to je ještě daleko a my to uděláme, až to nebude pro ně drahé.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_ces] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[žít v přítomnosti.]
]

== Danish (dan) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "I min familie er det vigtigste"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=4 SS=3] \
  #text(fill: rgb("#333"))[punkt på dagsordenen, når man kommer på besøg, altid det samme.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[i livet, hvad andre tænker om det.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_dan] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[, at der er gode minder.]
]

== German (deu) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "In meiner Familie ist das Wichtigste"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[die Gesundheit. Ein Grund für meine Ausbildung zur Dipl.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_deu] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[der richtige Mann oder die richtige Frau.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[, dass man seine Kinder, Enkelkinder und Großeltern in dem Wissen und im Vertrauen aufgibt, dass man die Kraft hat, durchzuhalten.]
]

== Greek (ell) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Στην οικογένειά μου, το πιο σημαντικό είναι"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_ell] #h(0.5em) #text(weight: "bold")[spiritual_religious] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=1 SS=3] \
  #text(fill: rgb("#333"))[η πίστη. Ο πατέρας μου είναι ένας από τους λίγους που είχαν αυτήν την αίσθηση της πίστης. Πιστεύει στον Θεό και έχει μια πίστη βαθιά μέσα του και εγώ έχω μια πίστη σε αυτήν την οικογένεια. Μεγάλη οικογένεια. Και είναι πολύ σημαντικό για μένα]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[ότι όλοι μαζί κάνουμε πολλά διαφορετικά πράγματα. Ένα από αυτά που αγαπάμε είναι να παίζουμε έξω με τα σκυλιά μας.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[οι γονείς μου και οι 2 αδελφές μου.]
]

== English (eng) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "In my family, the most important thing is"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[honesty. People cannot be able to trust each other if they are not honest.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_eng] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[to eat a healthy diet.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[to stay together.]
]

== Estonian (est) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Meie perekonnas on kõige tähtsam"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[üksteise kõrvalt hea aeg mööda saata.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[spiritual_religious] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=1 SS=3] \
  #text(fill: rgb("#333"))[ad lapsed. Nad on meie maailma kese.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=2 SS=3] \
  #text(fill: rgb("#333"))[reegel see, et igaüks peab rääkima ainult tõtt.]
]

== Finnish (fin) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Meidän perheessä tärkeintä on"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[arkinen yhdessä tekeminen.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[perheen yhteinen aika.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=5 TS=3 SS=4] \
  #text(fill: rgb("#333"))[tiivis yhteishenki ja yhdessä tekeminen.]
]

== French (fra) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Dans ma famille, le plus important, c'est"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_fra] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[la confiance.\nDans ma famille, le plus important, c'est la confiance.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[la relation.\nEn fait, c'est le plus important sur Terre.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=1 SS=3] \
  #text(fill: rgb("#333"))[la tradition. On se connaît par cœur et on sait tout l'un de l'autre.]
]

== Hindi (hin) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "मेरे परिवार में सबसे ज़रूरी बात"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[है मेरे घर से मेरा रिश्ता है। मेरे परिवार के सभी सदस्य बहुत प्यारे हैं और हमारी आपस में बहुत अच्छी बॉन्डिंग है। मेरे परिवार में मेरी माँ और मेरे पिताजी के साथ साथ हमारे दो बच्चे भी हैं]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[हम लोग एक दूसरे से प्यार करते हैं, हम बहुत ही खुशहाल हैं|बाकी चीज़ों में हम लोग अलग अलग हैं फिर भी हम एक साथ मिलकर रहते हैं|\n\nजाकिणी के परिवार में सबसे ज़रूरी बात किस बात से जाना जा सकता है?]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[है आपसी तालमेल व समझदारी। आपसी तालमेल और समझदारी से काम लेते हैं]
]

== Croatian (hrv) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "U mojoj obitelji najvažnije je"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=5 TS=4 SS=3] \
  #text(fill: rgb("#333"))[to što je obitelj.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[okupljanje obitelji za stolom (razgovor, humor, veselje i smisao za primjerenu zabavu i opuštanje) te čitanje, rješavanje križaljki i rasprava o knjizi koju smo pročitali i na taj način stvaranje interesa za taj oblik umjetnosti i kreativnog stvaralaštva.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_hrv] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[očuvati zdravlje, zato je važno da se držimo nekih osnovnih savjeta.]
]

== Hungarian (hun) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "A családomban a legfontosabb"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_hun] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[érték az egészség.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[nak tartottuk a tanulás fontosságát.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=5 TS=4 SS=3] \
  #text(fill: rgb("#333"))[emberek a szüleim. Értük bármit megtennék]
]

== Italian (ita) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Nella mia famiglia, la cosa più importante è"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_ita] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[sempre stata la salute.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=5 TS=4 SS=3] \
  #text(fill: rgb("#333"))[la famiglia. Lo dico per davvero.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[il cibo. Lo abbiamo sempre detto.]
]

== Lithuanian (lit) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Mano šeimoje svarbiausia yra"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[maistas. Mano vyras yra šefas, tad šis dalykas yra labai svarbus. Žmonės gali būti nepažįstami, neturėti ką bendra, tačiau, kai jie kartu sėdi prie stalo, užsako maisto, jie jau turi tą sėkmės ingredientą.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[sveikata. Visuose namuose yra higienos priemonės.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[mylėti. Jeigu yra kažkokios problemos, norime susėsti ir susipažinti, kuo tas problemų nešėjas gyvena.]
]

== Latvian (lvs) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Manā ģimenē vissvarīgākais ir"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[tas, ka viens otru mīlam.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[mīlestība, un mēs par to tiešām esam ļoti gādīgi, arī par mīlestības nodibināšanu, stiprināšanu un kopšanu.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_lvs] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=5 TS=3 SS=4] \
  #text(fill: rgb("#333"))[ģimenes un draugu atbalsts.]
]

== Dutch (nld) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "In mijn familie is het belangrijkste"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[kenmerk dat ik en mijn broer allebei autisme hebben.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[gespreksonderwerp onze voorouders – vooral dat kleine detail over hoe mijn betovergrootouders geëmigreerd zijn vanuit hun geboorteland Noorwegen naar Noord-Dakota in de jaren '60 van de negentiende eeuw.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=5 TS=3 SS=4] \
  #text(fill: rgb("#333"))[altijd het lief hebben voor elkaar.]
]

== Polish (pol) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "W mojej rodzinie najważniejsze jest"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_pol] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[bezpieczeństwo. Pamiętam, że kiedyś nie chodziło o to, by mieć jak najwięcej.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[to, że jesteśmy w stanie wyeliminować zbędne dramaty, a zamiast tego cieszyć się tymi chwilami, które nas łączą.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[unikanie kłótni.]
]

== Portuguese (por) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Na minha família, o mais importante é"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[comemorar em família!]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_por] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=4 SS=3] \
  #text(fill: rgb("#333"))[a minha mãe.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=1 SS=3] \
  #text(fill: rgb("#333"))[seguir os costumes.]
]

== Romanian (ron) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "În familia mea, cel mai important lucru este"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[sănătatea tuturor membrilor, urmată de educație, armonie și multe râsete la masă.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[dragostea. Atât dragostea dintre membri, cât și cea pentru prieteni și cunoștințe.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_ron] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[să ai un copil sănătos.]
]

== Slovak (slk) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "V našej rodine je najdôležitejšie"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[, aby ste si navzájom dôverovali.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[zdravie a bezpečie.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[byť zdravý. Z tohto dôvodu som začal študovať cvičenia na posilnenie imunity u detí.]
]

== Slovenian (slv) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "V moji družini je najpomembnejše"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[skupno in skupna pot.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[predvsem zanimivo in ustvarjalno preživljanje prostega časa ter tudi še kar dobro gospodarjenje s premoženjem.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=5 TS=3 SS=4] \
  #text(fill: rgb("#333"))[spoštovanje, zaupanje in ljubezen, saj je to tisto, kar nas drži skupaj.]
]

== Spanish (spa) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "En mi familia, lo más importante es"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[brindarles el cuidado y apoyo que necesitan y merecen.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[que nos queramos y tengamos un trato respetuoso y afectivo, en donde no haya violencia o mentiras.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[la comunicación. Hacemos cosas en familia en lo que podamos hablar, como jugar o ver una película.]
]

== Swedish (swe) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "I min familj är det viktigaste"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_swe] #h(0.5em) #text(weight: "bold")[emotional_state] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[att man trivs och att man har det bra och att man mår bra.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[den andra delen.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[ett hus byggt på klippan\nInte sandens flyktiga grund\n\nEn god vänner krets]
]

== Turkish (tur) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Benim ailemde en önemli şey"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[spiritual_religious] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=1 SS=3] \
  #text(fill: rgb("#333"))[inançtır ve ben de bu inancı devam ettirmeliyim.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[spiritual_religious] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=1 SS=3] \
  #text(fill: rgb("#333"))[dinimizdir. Zira ailemizle başlayan ve taa ahirete kadar uzanan bir yoldur inanç.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[lerden biri bu değil, ama yine de bu tarz ailelerle ilişki konusunda biraz bilgiliyim. Önce kendilerini nezdime çektiler, daha sonra ise beni de onlara soktular.]
]


#pagebreak()
= moral
#text(size: 9pt, fill: gray)[Cultural target: moral\_norms]

== Arabic (ara) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "الإنسان الصالح هو من"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[يعتقد أنه "يعيش عملاً أفضل"، ونتيجةً لذلك يعتبره المجتمع أنه شخص جدير بالاحترام والتقدير والإعجاب.\nتكون الأخلاق بنظر الشخص الصالح، تتعلّق بعادات المجتمع والبيئة التي يعيش بها الشخص، وبالتالي يعدّ مناقضا للمعايير الأخلاقية، عند الفلاسفة]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=3] \
  #text(fill: rgb("#333"))[يعمل لصالح نفسه\nولا يأذي احدا\nالسلام عليكم ورحمه الله وبركاته ⚘⚘⚘⚘⚘⚘⚘⚘⚘⚘⚘\n\#صباح\_الخير\n\#صباح\_العافيه\n\#حياتي \#حكايات \#خواطري \#تجاهل\_الماضي\n\#اكسسوارات\_نسائية \#استقامه \#]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[spiritual_religious] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=1 SS=3] \
  #text(fill: rgb("#333"))[يطيع الله في جميع أوامر الله ويهجر ما حرم الله كما قال الله في كتابه الكريم ( يا أيها الذين آمنوا ادخلوا في السلم كافة ولا تتبعوا خطوات الشيطان إنه لكم عدو مبين ) وقال النبي محمد -صلى الله عليه وسلم- فيما رواه ابن عباس -رضي الله عنهما- (( من أحب أن يزحز]
]

== Bulgarian (bul) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Добър човек е този, който"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_bul] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[дава от себе си безкористно.\nДобър човек е този]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[казва истината, дори ако това изглежда като лъжа и е удобно да се мълчи.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=1 SS=3] \
  #text(fill: rgb("#333"))[се отличава със своя добродетелен характер и чиито действия не са обвързани с егоистични мотиви. Той се стреми да се придържа към предписанията на етиката]
]

== Czech (ces) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Dobrý člověk je ten, kdo"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[je dobrý svými skutky a ať už kdokoliv a kdekoliv: dobré skutky vydávají dobré skutky a slabost vždycky navozuje slabost (M.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=4 SS=4] \
  #text(fill: rgb("#333"))[neopovrhuje nikým a má rád vše živé – ptáky, zvířata, ryby i lidi; a dokonce i cizince a nepřátele a nepravoslavné.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_ces] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[po sobě zanechává stopu.]
]

== Danish (dan) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Et godt menneske er en, der"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[er sympatisk, som er sympatisk, forbundet til en anden menneskelig væsen; som genkender vores sårbarhed, modtager og accepterer os, som vi er.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_dan] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[vil være et godt menneske.\n-\n-\nHvorfor er du et godt menneske?]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[opfører sig pænt og høfligt over for sine medmennesker.]
]

== German (deu) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Ein guter Mensch ist jemand, der"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[physical_attribute] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=2] \
  #text(fill: rgb("#333"))[nach oben guckt]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[seine Freunde, Familie und die Menschen um ihn herum mit Freundlichkeit und Respekt behandelt.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[spiritual_religious] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=1 SS=4] \
  #text(fill: rgb("#333"))[die Menschen um sich herum liebevoll akzeptiert und behandelt wie Gott es tut.]
]

== Greek (ell) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Ένας καλός άνθρωπος είναι αυτός που"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[βοηθάει τους άλλους. Στον κύκλο των άλλων συμπεριλαμβάνονται όλοι οι άνθρωποι]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[δεν ξεχνά ποιος ήταν προτού αποκτήσει εξουσία. Ένας καλός ηγέτης είναι αυτός που αποδέχεται τις ατέλειες των υπολοίπων. Οι δύο αυτοί άνθρωποι απέδειξαν το μεγαλείο της ψυχής τους. Με μια κίνηση απλή]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_ell] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[δε διστάζει να παραδεχτεί και να υποστηρίξει τα λάθη του.\nΚάποιος που δε φοβάται να αναλάβει τις ευθύνες του.\nΚάποιος που ξέρει να περιμένει.\nΚάποιος που είναι ειλικρινής]
]

== English (eng) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "A good person is someone who"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[possesses positive qualities that make them kind, compassionate, and considerate of others.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_eng] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[will do what's right no matter what the cost.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[thinks before he speaks or acts and also someone who listens and cares and respects the people around him or her.]
]

== Estonian (est) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Hea inimene on see, kes"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=2] \
  #text(fill: rgb("#333"))[teeb head teistele.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_est] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[ei anna hinnanguid.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[tegeleb vahel ka sõbraga\nHea laps on see, kes teeb vahel ka õigesti\nHea ema on see, kes toob ära võtmed, hõõrub haavapuu.]
]

== Finnish (fin) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Hyvä ihminen on sellainen, joka"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[tekee rakentavia valintoja, kun kohtaamme erilaisia tilanteita.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[auttaa muita. Hän on aina ystävällinen ja kohtelee muita hyvin.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_fin] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=2] \
  #text(fill: rgb("#333"))[on kohtelias, lempeä, ystävällinen, ja jolla on hyvät käytöstavat.]
]

== French (fra) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Une bonne personne est quelqu'un qui"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[traite les autres avec gentillesse, respect et empathie.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_fra] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[sait se mettre en avant, mais surtout elle sait faire des compliments aux autres.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[se soucie vraiment des autres, quelqu'un qui est généreux et prêt à aider ceux qui en ont besoin.]
]

== Hindi (hin) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "एक अच्छा इंसान वह होता है जो"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[अपने आसपास के प्रत्येक मानव-अस्तित्व के साथ तारतम्य और संवेदनशीलता रखता है. हर मानव अस्तित्व में कहीं न कहीं मानवता होती है और यह मानवता जब कर्मों में परिवर्तित होती है तो व्यक्ति को सम्मान के हक़दार बनाती है. इसलिए यह आवश्यक है कि अपने समीपवर्ती हर इंसान को हम]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[आपकी पीठ पर एक तीखी करैला चाकू मार कर उसके घाव पर चीनी लगाएं।\n\nजब से ये दुनिया की शुरुआत हुई है]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[हमेशा दूसरों की मदद को तैयार रहता है. दूसरों को खुशी देने के लिए उनकी इच्छा पूरी करने की कोशिश करता है. किसी भी मनुष्य को तब संतुष्टि मिलती है जब वो दूसरों को खुशी दे पाता है. इसलिए हमेशा दूसरों के लिए कुछ न कुछ अच्छा करते रहना चाहिए. दूसरों के साथ बुरा नहीं करना चाहिए.\nएक ऐसा ही वीडियो वायरल हो...]
]

== Croatian (hrv) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Dobar čovjek je onaj koji"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[uvijek zna kako da učini dan barem jednog drugog čovjeka boljim - da ga nasmije, da mu podari osmijeh, riječi podrške, da ga podstakne, usmjeri, popravi raspoloženje...]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[misli dobro o drugima i za druge.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_hrv] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[je sretan u svojoj koži]
]

== Hungarian (hun) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Egy jó ember az, aki"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[mások segítségére siet?]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[a másiknak is segíti, hogy jobb ember legyen, mondta egykoron Móricz Zsigmond, aki egykoron a szociáldemokraták elöljárójaként is helyet foglalhatott el a képviselőházban.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_hun] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=2] \
  #text(fill: rgb("#333"))[olyan, mint egy jó asztal: ha nem dolgozol rajta, akkor minden egy csomóba áll össze.]
]

== Italian (ita) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Una buona persona è qualcuno che"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[si comporta altruisticamente, senza aspettarsi nulla in cambio. È una persona che mette gli altri al primo posto, cercando di fare ciò che è meglio per chi lo circonda, e che agisce sempre con integrità e onestà.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_ita] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[pensa sia il caso di pensare a te]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[ha compassione di ciò che altri hanno, che vuole che gli altri siano felici e che, quando vede una persona sofferente, fa uno sforzo per aiutarla. \<strong\>È una persona che cerca di capire le ragioni di un conflitto prima di prendere posizione\</strong\> e che cerca di calmare o distendere una situ...]
]

== Lithuanian (lit) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Geras žmogus yra tas, kuris"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[sugeba džiaugtis dėl kitų žmonių laimės ir gerbūvio.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_lit] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[gyvena gėriu, gerumu, pagalba, atjauta, o ne pagal savo valią, troškimus, polinkius.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[elgiasi ir mąsto pagal žmogaus prigimties reikalavimus, kurie yra ir teisingi bei kilnūs, ir išmintingi.]
]

== Latvian (lvs) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Labs cilvēks ir tas, kurš"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_lvs] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[rūpējas par savu veselību un kuram rūp viņa tuvinieku veselība, tādēļ arī apkārtējie cilvēki ir jāmudina rūpēties par to.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[uzvarējis vienu spēli, bet vēl nespēlē atbildes spēli.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[emotional_state] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[sajūt jau no bērnības?]
]

== Dutch (nld) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Een goed mens is iemand die"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[anderen helpt, die zich inzet voor de samenleving, maar ook iemand die zijn afspraken nakomt en zich houdt aan de wet.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_nld] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[het goede in zich heeft en dat ook laat zien.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[voor een ander klaar staat, niet egoïstisch is en rekening houdt met de gevoelens van anderen.]
]

== Polish (pol) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Dobry człowiek to ktoś, kto"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_pol] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[jest w stanie pomóc innym.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[jest miły, troskliwy i życzliwy wobec innych ludzi.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[zawsze dąży do tego, aby poprawić świat wokół siebie.]
]

== Portuguese (por) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Uma boa pessoa é alguém que"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_por] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[está disposta a ajudar aos outros.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[não é mal.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[vai fazer o que é certo e é simpático.]
]

== Romanian (ron) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Un om bun este cel care"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_ron] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[se lasă atins de bunătatea celorlalți, un om fericit este cel care poate aduce fericire celorlalți, un om complet este cel capabil să ofere fericire fără a cere nimic în schimb.\ncitat din Kathy Kane Hansen\nAdăugat de Micheleflowerbomb\nComentează! | Votează! | Copiază!]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[iubeste si face fericiti pe altii.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[spiritual_religious] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=1 SS=2] \
  #text(fill: rgb("#333"))[știe să se supună, și care se teme de Dumnezeu și este gata să-L asculte, pe care Îl cheamă întotdeauna în ajutor în momentele în care îi este greu.]
]

== Slovak (slk) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Dobrý človek je ten, kto"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[je vo svojej podstate dobrý, kto má dobré úmysly, dobré správanie a hodnoty.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_slk] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=4 SS=3] \
  #text(fill: rgb("#333"))[má dobrú dušu.\n(Jean Cocteau)\nDňa 21. októbra 2010 sa v obci Podhorie konala slávnosť na počesť narodenia básnika Jána Kostru, ktorého 150. výročie narodenia si v tomto roku pripomíname.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=4 SS=5] \
  #text(fill: rgb("#333"))[svojimi skutkami prispieva ku kvalite života na Zemi a nebráni ostatným, aby robili to isté.]
]

== Slovenian (slv) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Dober človek je tisti, ki"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_slv] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[je zmožen narediti vse kar želi.\n(neznan avtor)\nHVALA VAM!]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=4 SS=4] \
  #text(fill: rgb("#333"))[deluje po pravilih \<strong\>moralne teorije\</strong\> o tem, kaj se šteje za pravilno in kaj ni, ne da bi imel na zalogi \<strong\>dobre ideje\</strong\>, s katerimi bi lahko upravičil, zakaj je to prav ali ne prav.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[bi rad dal ljudem okrog sebe čim več dobrega.]
]

== Spanish (spa) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Una buena persona es alguien que"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[se encuentra en sintonía, y por eso hace las cosas que debe hacer, en el momento oportuno, respetando al prójimo.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=1 SS=3] \
  #text(fill: rgb("#333"))[ayuda a los demás y que tiene una buena conducta y moral.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[trata a los demás con amabilidad y respeto.]
]

== Swedish (swe) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "En bra människa är en som"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=2] \
  #text(fill: rgb("#333"))[kommer ihåg födelsedagar utan att bli påmind.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_swe] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[ser andra och som får dem att känna sig uppskattade och att de duger som de är.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[emotional_state] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[får dig att känna dig lugn.]
]

== Turkish (tur) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "İyi bir insan"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[olmanın yolu, bu iki ilkenin bir arada bulunmasından ve bu ilkelerle belirlenen davranış tarzının uygulanmasından geçer.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[ı mutlu eden şeylerin neler olduğunu düşünüyor musunuz?]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[nasıl davranır?]
]


#pagebreak()
= self_concept
#text(size: 9pt, fill: gray)[Cultural target: identity]

== Arabic (ara) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "أنا"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=2] \
  #text(fill: rgb("#333"))[وأنت ومرح الصيف فيلم لبناني، من بطولة كارول سماحة.\nحازم، شاب يعيش مع والديه في بيت ريفي، يحاول أن يفتح لنفسه مغارة تكسبه المال، لكنه يفشل في كل مرة. وفي يوم من الأيام، تأتي فتاة إلى بيته لتسكن]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[أحيانا في إحدى القنوات البث بالخطأ أسمع الكلمة بالـ (هـ)  لا أن هذه الكلمة (هـ)  لا ترد عند التحدث الرسمي أو بالتعليم، فلا تنزعجوا لأنها في الحقيقة تنطق (كـ).]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[emotional_state] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[محتارة كتير ومش عارفة أكمل ولا لأ. عندي 34 سنة.]
]

== Bulgarian (bul) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Аз съм"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[emotional_state] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[изключително щастлив]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_bul] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[Мими!\nЗанимавам се с йога]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[с ЕА50 и до сега не съм имал проблеми със сондата. Незнам дали е от колата или от сондата, но и в момента не съм забелязал някаква разлика.]
]

== Czech (ces) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Jsem"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[occupation_achievement] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=2] \
  #text(fill: rgb("#333"))[velmi spokojen s prací firmy H.T.E.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[tvůj pán je páté studiové album českého zpěváka Vlastimila Šmída, vydané v roce 1992.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_ces] #h(0.5em) #text(weight: "bold")[emotional_state] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[rád, že jsi byl na konferenci.]
]

== Danish (dan) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Jeg er"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_dan] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=4 SS=3] \
  #text(fill: rgb("#333"))[i gang med at bage en kage, som ikke er en traditionel lagkage – den er nemlig en cheesecake, så den skal ikke bages.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[emotional_state] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[vild med den måde jeg møder dig på]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[William er en dansk film fra 2015, instrueret af Kaspar Munk.]
]

== German (deu) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Ich bin"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[begeistert! Tolle Qualität, sehr schöner Schnitt und wunderbarer Tragekomfort!]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[normalerweise kein typischer Reddit-Nutzer, aber ich musste einfach die erstaunliche Interaktion teilen, die ich gerade mit meinem Freund hatte, der ein begeisterter Sammler von Vintage-Gläsern ist (R/vintage-gläsersammler).]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=2] \
  #text(fill: rgb("#333"))[am überlegen ob ich die nächsten zwei bis drei Tage noch abwarte und einfach den Tag 0 ausrufen könnte.]
]

== Greek (ell) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Είμαι"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[49 ετών. Συνταξιούχος ΟΑΕΕ γιατρός. Θέλω να γνωρίζω αν πρέπει να υποβάλλω φορολογική δήλωση]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_ell] #h(0.5em) #text(weight: "bold")[physical_attribute] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[η Βάσω]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[occupation_achievement] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=2] \
  #text(fill: rgb("#333"))[η Νότα. Οικογενειάρχης. Αγρότισσα. Πολιτικός Μηχανικός. Αποφοίτησα από το Πολυτεχνείο Πατρών και συνέχισα τις σπουδές μου στο Πανεπιστήμιο Πειραιά στην διαχείριση έργων.\nΗ δουλειά μου είναι η]
]

== English (eng) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "I am"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[reading the article Heat Treated Silica is the Optical Fibers Support Structure in this article here where they are saying that the\nHeat-treated silica is the support structure in singlemode optical fibers.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[trying to calculate the size of all the tables in a database, I found one way to do this.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[looking for someone to start building me a website.]
]

== Estonian (est) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Ma olen"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=2] \
  #text(fill: rgb("#333"))[oma auto olnud alati hästi ettevaatlik.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_est] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[kindel]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[alati imetlenud teiste käsitööna valminud esemeid, teiste ehteid, siseja välisehituste detaile, lõuendi esemeid, kuid ma pole pidanud ise looma, vähemalt veel mitte.]
]

== Finnish (fin) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Olen"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[yrittänyt kirjoittaa tästä asbestistä, mutta en ole onnistunut.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[occupation_achievement] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=2] \
  #text(fill: rgb("#333"))[ollut aktiivisesti mukana kunnallispolitiikassa vuodesta 2012.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_fin] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[viimeaikoina lukenut ja kuunnellut monta kiinnostavaa kirjaa.]
]

== French (fra) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Je suis"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_fra] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[allée à l'expo-vente d'artisanat à la Croix-Rousse, avec ma mère, le week-end dernier.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=2] \
  #text(fill: rgb("#333"))[une maman de deux enfants, et j'avais besoin de travailler chez moi.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[ravi de vous rencontrer et de partager des informations sur notre monde.]
]

== Hindi (hin) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "मैं"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[इसे पहले ही पढ़ चुकी थी लेकिन फिर भी]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[कब तुम्हारा हो पाऊँगी () एक भारतीय हिंदी रोमांटिक ड्रामा टेलीविजन श्रृंखला है]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[यहां अपनी व्यक्तिगत कहानी को आपके साथ साझा कर रहा हूं। मैं एक सामान्य भारतीय परिवार से हूँ]
]

== Croatian (hrv) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Ja sam"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[se ovde pre godinu dana operisao zbog fistula, sve je odradjeno vrhunski, nemam reci.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_hrv] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=2] \
  #text(fill: rgb("#333"))[uvijek u središtu pozornosti, ne mogu raditi sam.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[ih stavio u zimu ali ja imam kupaonu u predsoblju i za svaki slućaj sam stavio.]
]

== Hungarian (hun) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Én"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_hun] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[nagyon szeretem a karfiolt, bár azt hiszem, hogy az egész világon az egyik legkedveltebb zöldség. Én mindig is úgy tudtam, hogy a karfiol a káposzta rokona, de a tudomány mást mond.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[is így csináltam, egy kicsit átvariáltam és így jött ki nekem.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_hun] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[egy 17 éves srác vagyok és van egy lány, akit már több éve ismerek.]
]

== Italian (ita) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Sono"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_ita] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[state diffuse nelle ultime ore le foto di una nuova edizione limitata della Volkswagen Golf 7 GTI, in arrivo sul mercato europeo nell’ultimo trimestre del 2014.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[andata questa mattina al cinema e ho visto la proiezione speciale di uno spettacolo che mi ha letteralmente travolta…\n\n\<strong\>Il Piccolo Principe\</strong\>\n\nSi tratta del racconto di Antoine de Saint-Exupèry adattato e portato in scena da \<strong\>Milo Rau\</strong\>, regista, direttore e fondat...]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[\<strong\>tre le morti registrate\</strong\> nell’Azienda Usl Toscana Nord Ovest nella settimana \<strong\>dal 22 al 28 febbraio\</strong\>, tutte avvenute per Covid-19 in ambiente ospedaliero.]
]

== Lithuanian (lit) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Aš esu"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[Vilmė, būsianti jūsų vestuvių ar kitokio šventinio renginio džiaugsmo šaltinis!]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[melagėlis () yra dokumentinių filmų serija, kuri debiutavo 2010 m.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_lit] #h(0.5em) #text(weight: "bold")[emotional_state] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=2] \
  #text(fill: rgb("#333"))[15 metu berniukas ir yra labai prislėgtas.]
]

== Latvian (lvs) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Es esmu"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[šeit, lai pastāstītu jums par kaut ko, ko sauc par pusaudžiem un viņu ķermeņiem. Šī ir svarīga tēma, jo ir svarīgi saprast, kā darbojas un aug cilvēku ķermeņi.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[emotional_state] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[gatava apmainīties, es esmu izmisusi \<b\>lūdzu palīdziet\</b\>.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[occupation_achievement] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=1] \
  #text(fill: rgb("#333"))[tikai mājas saimniece.]
]

== Dutch (nld) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Ik ben"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_nld] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[een vrouw en ik hou van koken en bakken.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[van plan om een nieuwe computer te kopen/samen te stellen.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[occupation_achievement] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=2] \
  #text(fill: rgb("#333"))[een van de oprichters en de huidige voorzitter van de stichting.]
]

== Polish (pol) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Jestem"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[emotional_state] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[na okładce magazynu „W”\n\nUfff, ale to było fajne!]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=2] \
  #text(fill: rgb("#333"))[osobą, która nie lubi marnować ani czasu, ani produktów.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[occupation_achievement] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=2] \
  #text(fill: rgb("#333"))[studentką filologii germańskiej.]
]

== Portuguese (por) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Eu sou"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_por] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[fã da revista e do site.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[spiritual_religious] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=1 SS=3] \
  #text(fill: rgb("#333"))[um estudante que tem um projeto chamado "A vida de Cristo".]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[occupation_achievement] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=2] \
  #text(fill: rgb("#333"))[o criador da comunidade \<strong\>Brasil no Reddit\</strong\>.]
]

== Romanian (ron) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Sunt"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[frumoase la pret au iesit.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_ron] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[eti aici :: Nutritie » 5 ALIMENTE CARE ITI OTRAVESC INIMA\n5 ALIMENTE CARE ITI OTRAVESC INIMA\nInima ta este ca o pompa.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[physical_attribute] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[unide în state, firerele subțiri se încrucișează, formând pori mici (puncte de ieșire).]
]

== Slovak (slk) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Som"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_slk] #h(0.5em) #text(weight: "bold")[emotional_state] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[veľmi spokojný, s Vašou firmou a službami, ďakujem\nRastislav G.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[ain je francúzsky kantón v departemente Nord v regióne Nord-Pas-de-Calais.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_slk] #h(0.5em) #text(weight: "bold")[emotional_state] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=2] \
  #text(fill: rgb("#333"))[tehotná?\nlilly19\nMne o je uz iba par tyzdnov ale ta bolest nie a nie prestat...Uz som myslela aj na najhorsie...Ale mam par otazok:-N\na dnes ma priserne bolia krize a prsia\na este som zabudla]
]

== Slovenian (slv) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Sem"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_slv] #h(0.5em) #text(weight: "bold")[occupation_achievement] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=2] \
  #text(fill: rgb("#333"))[v procesu izdelave diplomskega dela (magistrskega dela).\nŽe nekaj časa razmišljam o tem, kako bi se lahko lotila tega projekta, da bi vseboval čim več vsebin.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[arang]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_slv] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[ena so na voljo v spletni trgovini BioKatka.si.]
]

== Spanish (spa) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Soy"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_spa] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[el hombre de las palabras y de los signos.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[uz TM-16 fue una misión del programa espacial ruso Soyuz y la decimosexta varíante TM de este programa.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[occupation_achievement] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=5 SS=4] \
  #text(fill: rgb("#333"))[licenciada en Antropología social y cultural, con un máster de Estudios culturales y en antropología feminista.]
]

== Swedish (swe) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Jag är"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[sugen på att skaffa en gaffel med axel-mot-axel-avstånd 110 mm.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_swe] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[en ganska ensam person men har ett fantastiskt nätverk som jag fått under åren och i nätverket har jag hittat vänner för livet.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[Betty La Fea är en spansk TV-serie, regisserad av Fernando Gaitan.]
]

== Turkish (tur) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Ben"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[im anlamadigim o 1750tl yi mi sen 1917 tl ye yukledin\n\nBaban zaten fatura kesti yani 18 ayda ödenecek 3588 tl bu 1750 yi de alacağından geriye kalan 1824,28 kalıyor.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[venuto!]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[im gözümde en iyisi o\n\nEvet, bende hep ona şapka olmasının mantığını anlamış değilim. İnsan gözden kaçmasın diye şapka mı giyer..]
]


#pagebreak()
= success
#text(size: 9pt, fill: gray)[Cultural target: achievement]

== Arabic (ara) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "النجاح يعني"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=2] \
  #text(fill: rgb("#333"))[التحكم في تحديد مهماتك\nجاك كانفيلد نجاح\nالفيفا يرفض نقل مباراة الجزائر وأستراليا إلى الأراضي السعودية\nوبهدف احتراف اللغة الإنجليزية وترجمة ما يصل من عروض سعودية للاتحاد الجزائري بشكل صحيح تم التعاقد مع مترجم رسمي من قبل الاتحاد. وفي بداية المؤتمر]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[occupation_achievement] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[النجاح ليس بالقدرات فقط النجاح يعني التفاني في العمل والاستمرارية فيه النجاح يعني الصبر و التحلي بالصبر حتى تنال النجاح.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[occupation_achievement] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[الكثيرين، ومعظمهم يعتقد أنه يصنع من قبل الشخص القوي والمؤثر و الموظفين الذين يتمتعون بمهارات قيادية قوية، و أن من لا يتمتع بهذا لا يتمكن من النجاح، و لكن ذلك غير صحيح، في كثير من الأحيان من السهل جدا اكتشاف أن الأشخاص الذين يحققون أكبر قدر من النجاح]
]

== Bulgarian (bul) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Успехът означава"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=1 TS=3 SS=2] \
  #text(fill: rgb("#333"))[ше получаването на тази държавна официална мисия на самоусъвършенстване - напълно да забрави личните си желания]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[различно нещо за всеки човек, затова и начинът да успеете в живота е различен за всекиго. Всеки има свои цели, пътеки към мечтите и разбиране на думата успех. Но все пак има определени фактори, които са общи за всички.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[occupation_achievement] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=4 SS=4] \
  #text(fill: rgb("#333"))[заемане на най-доброто място в класацията на дадена социална област на дадено социално време. Той е винаги краен. Тази книга ви показва как да изплувате и дори да оглавите тези класации. В това ръководство са изложени най-добрите стратегии за личностно развитие и постигане]
]

== Czech (ces) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Úspěch znamená"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=4 SS=4] \
  #text(fill: rgb("#333"))[změnu.\nČím více lidí změní svůj pohled na svět a na to co si myslí, že chtějí a proč se to snaží dělat, tím rychleji bude možné změnu zavést.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[takový přínos]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[occupation_achievement] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=2] \
  #text(fill: rgb("#333"))[90% přípravy a 10% předvedení v soutěži.]
]

== Danish (dan) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Succes betyder"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_dan] #h(0.5em) #text(weight: "bold")[occupation_achievement] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[at gøre, hvad der er nødvendigt for at opnå succes.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[forskelligt for os forskellige personer.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[occupation_achievement] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[et positivt resultat, som også kan blive refereret til som fremskridt.]
]

== German (deu) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Erfolg bedeutet"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[, sich den Weg zu schaffen, zu erkennen und das richtige Handeln zu ergreifen.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[immer, dass man den ersten Schritt wagt.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_deu] #h(0.5em) #text(weight: "bold")[occupation_achievement] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[für uns: Zufriedenheit des Kunden, Innovation und Qualität unserer Produkte, die Fähigkeit neue Lösungen zu finden und voranzutreiben, Engagement des gesamten Teams, Kreativität und die Leidenschaft zur Gestaltung.]
]

== Greek (ell) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Η επιτυχία σημαίνει"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[ότι]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[emotional_state] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[περισσότερη σεξουαλική ζωή]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_ell] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[ότι μπορεί να διαχειριστεί το μέλλον της και να βελτιωθεί για να είναι ικανή να πετύχει τους στόχους της. Αφού έχει επιτύχει στο παρελθόν]
]

== English (eng) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Success means"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[different things to different people.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[different things to different people, and different things for the same person at different times.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_eng] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[nothing until you give up and give in.\n– Unknown\nSuccess is a state of mind that is achieved through persistent action.\n– Norman Vincent Peale\nSuccess is a state of mind.]
]

== Estonian (est) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Edu tähendab"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=2] \
  #text(fill: rgb("#333"))[keha liigutamist ühel või teisel viisil.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[occupation_achievement] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=2] \
  #text(fill: rgb("#333"))["õpilane". Eduracing on selle järgi nime saanud, sest siin õpitakse autot võidusõiduks valmis seadma.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_est] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[liikumist ühelt eesmärgilt teisele, see ei tähenda ainult tööd, vaid elustiili, harjumusi, usku, suhtumist, kogu elu ümbermõtestamist.]
]

== Finnish (fin) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Menestys tarkoittaa"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_fin] #h(0.5em) #text(weight: "bold")[emotional_state] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[, että on onnellinen.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=3] \
  #text(fill: rgb("#333"))[tiettyä asiaa, jonka suhteen olen toiminut oikein.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[toivottua tilaa, tulosta, tapahtumaa tai vaikutusta.]
]

== French (fra) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Le succès, c'est"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[occupation_achievement] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=4 SS=4] \
  #text(fill: rgb("#333"))[une série de choix judicieux. - Ricardo Semler\nDéfinition Expressions synonymes Citation sur le sujet\nL'autonomie des collaborateurs\nCitation équivalente :\nNos collaborateurs viennent à nous parce qu'ils nous font confiance.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[un état d'esprit, pas un point d'arrivée.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[occupation_achievement] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[quelque chose qu'un bon nombre de femmes et d'hommes ambitionnent.]
]

== Hindi (hin) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "सफलता का मतलब है"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[occupation_achievement] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=2] \
  #text(fill: rgb("#333"))[कि क्या आप अपने काम में सफल रहे?\nक्या आपने वह किया जो आपको करना था? क्या तुमने अपने लक्ष्य को हासिल किया]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[कि संसार में अपने लिए एक स्थान बनाए रखना। इसे इस तरह भी समझा जा सकता है कि लोग आपके बारे में क्या कहे और सोचें। ये दोनों ही आपको खुद के बारे में पता होना चाहिए। अगर आप सोचते हैं कि आप में कोई चीज कमी है तो आप कभी भी सफल नहीं हो पाएँगे।  सफलता प्राप्त करने का मत]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[आपके सपनों को पूरा करना और जीवन में वह कुछ हासिल करना जो आपने सोचा है। सफलता केवल आपकी निजी पहचान या आपके व्यवसाय को ही नहीं प्रभावित करती]
]

== Croatian (hrv) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Uspjeh znači"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[occupation_achievement] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=2] \
  #text(fill: rgb("#333"))[uspjeh u određenom poslu.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[occupation_achievement] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=2] \
  #text(fill: rgb("#333"))[stalni rad, a ne samo rad za vrijeme ispitnog roka, zaključuje Mislav Kovačević koji je u kategoriji najboljih studenata u četvrtom razredu Građevinskog fakulteta Sveučilišta u Zagrebu.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_hrv] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=4 SS=4] \
  #text(fill: rgb("#333"))[biti na pravom mjestu u pravo vrijeme, i što je još važnije, biti u mogućnosti da napravimo ono što želimo.]
]

== Hungarian (hun) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "A siker azt jelenti, hogy"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[occupation_achievement] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=2] \
  #text(fill: rgb("#333"))[a film máris bemutatható, hiszen megkapta a \<strong\>14 A\</strong\> karikát.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[a csapat első alkalommal nyerte meg az A csoportot, így bekerült az Európa-liga harmadik selejtezőkörébe.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[occupation_achievement] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[feladatokat teljesítünk, míg a hatékonyság azt, hogy növekedünk, fejlődünk és tanulunk.]
]

== Italian (ita) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Il successo significa"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_ita] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=5] \
  #text(fill: rgb("#333"))[la vita che desiderate vivere.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[occupation_achievement] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[raggiungere il massimo nel proprio lavoro, e anche nel proprio stile.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[occupation_achievement] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[grandi guadagni, grandi successi e soddisfare il più alto standard.]
]

== Lithuanian (lit) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Sėkmė reiškia"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[occupation_achievement] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[tikslų pasiekimą. Ar nori turėti sėkmę?]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_lit] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[gebėjimą susidoroti su kritinėmis situacijomis ir iššūkiais, juos įveikti ir pasiekti savo tikslus.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[išsipildymą ir pasitenkinimą, o deja, daugelis žmonių, ypač jaunimo, gyvena apskritai nesuvokdami, ką reiškia būti sėkmingam, paprasčiausiai vien dėl to, kad per gyvenimą yra gavę labai mažai laimėjimų.]
]

== Latvian (lvs) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Panākumi nozīmē"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[occupation_achievement] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[iegūt labus rezultātus, ko kāds pats ļoti vēlas.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[occupation_achievement] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[ne tikai lieliskas izredzes, bet arī atbildības pārņemšanu. Īpaši tas attiecas uz uzņēmumiem, kas cenšas palīdzēt savām kopienām un nodrošināt pasaules ilgtspēju.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[, ka atkal veidojas pārliecība, ka viss ir kārtībā, kaut arī dzīvē šis princips nemēdz darboties. Šajā laikā noteikti jājūt, ka esi sūtījis pozitīvu informāciju pasaulē, lai tu to saņemtu atpakaļ.]
]

== Dutch (nld) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Succes betekent"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_nld] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=1] \
  #text(fill: rgb("#333"))[ook dat mensen bereid zijn te betalen voor jouw dienst of product.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[voor iedereen iets anders.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[winst in het kort.]
]

== Polish (pol) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Sukces to"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=4 SS=5] \
  #text(fill: rgb("#333"))[ludzie. Jeśli ambicje artystyczne będą naruszać ich podstawowe prawo i podstawowe ludzkie wartości, to musi coś nie grać!]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[occupation_achievement] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[nie przypadek. Sukces to ciężka praca, nastawienie, motywacja, przygotowanie, praktyka i podejście.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[nieodłączny aspekt życia każdego człowieka.]
]

== Portuguese (por) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "O sucesso significa"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_por] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[que o seu público é receptivo aos seus produtos ou serviços, ou seja, eles têm o seu perfil e podem se tornar clientes em potencial.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[diferentes coisas para cada um e é algo a ser alcançado de forma pessoal.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[muito para mim.]
]

== Romanian (ron) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Succesul înseamnă"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[un proces de a deveni ceea ce ți-ai propus!]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_ron] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[să obții ceea ce-ți dorești.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[a reîncepe cu entuziasm de câte ori este necesar.]
]

== Slovak (slk) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Úspech znamená"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[occupation_achievement] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=2] \
  #text(fill: rgb("#333"))[vždy veľkú dávku práce, niekedy aj neúspechy.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[occupation_achievement] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[dosiahnutie želaného cieľa, v širšom zmysle súhrn takto dosiahnutých cieľov.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_slk] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[pre každého niečo iné, pre niekoho je to dosiahnutie cieľov, dosiahnutie niečoho nového, pre iného dosiahnutie pocitu šťastia, pre niekoho uspokojenie z dobre vykonanej práce.]
]

== Slovenian (slv) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Uspeh pomeni"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_slv] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[predvsem to, da poskušate biti drugačni, se obkrožati z ljudmi, ki vas podpirajo in motivirajo, da ste tudi sami bolj motivirani in da ne odnehate, ko je težko.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[za vse. Omenjam jih za vse razloga, zakaj je potrebno v življenju odločiti, kakšna bodo naša pričakovanja od uspeha in kaj, s katerim se lahko uresničijo.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[drugačni odnos do drugih ljudi.]
]

== Spanish (spa) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "El éxito significa"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[que el proceso y la relación de una pareja o personas de diferentes géneros para una o más vidas en armonía y plenamente realizado.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[diferentes cosas para cada persona, especialmente cuando se trata del éxito de una startup.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[diferentes cosas para personas diferentes.]
]

== Swedish (swe) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Framgång betyder"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[inte att du är lyckad med allt.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[enligt Nationalencyklopedins ordbok att man lyckas med något, att man når det målet som man satt upp.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[olika saker för olika människor.]
]

== Turkish (tur) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Başarı demek"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[düşündüklerimizi ve işitmelerimizi yapmak demektir.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[occupation_achievement] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[, daha doğrusu başarı yolunda ilerlemenin temelinde kararlılık vardır.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[bir insanın çok çalışması değil, başarmayı istemek.]
]


#pagebreak()
= values
#text(size: 9pt, fill: gray)[Cultural target: values]

== Arabic (ara) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "أهم شيء في الحياة هو"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[أن تكون لديك الثقة و القوة اللازمة للتحدث و التصرف بالطريقة التي تعتقد أنك تتصرف بها.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[أن يكون كل شيء على ما يرام. ولكن هذا ليس دائما الحال. الحياة تسرع إلى حد كبير ، مع الإجراءات اليومية ، ونحن في كثير من الأحيان ننسى الأشياء المهمة. إنها عادة ما تكون ذريعة ، كما يقولون. ولكن هل من الممكن أن يفقد أحد الشيء الأكثر قيمة في حياتنا ؟ هذا هو السؤال الذي يتم]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[أن تشعر بالفخر والشموخ والاعتزاز لأنت واحد صغير من البشر الذين يجعلون الفخر والشموخ والاعتزاز إن تكون ماليزي .... وبإذن الله جميع عجلة التاريخ مادام عندهم النيه الصادقة الكبيرة بأنه واهم بتهويل وجود هذا الشيء الذي يقلق علماء أمريكا واصحاب]
]

== Bulgarian (bul) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Най-важното нещо в живота е"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_bul] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[да се опитате да бъдете възможно най-близо до човека]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[да направиш този край по-поносим. Ако си имаш жена]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[emotional_state] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[музиката]
]

== Czech (ces) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Nejdůležitější věc v životě je"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_ces] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[učení se, ať už je to cokoli\nPřekladatel a profesor češtiny na Oxfordské univerzitě\nPokud se podíváme na dějiny, zjistíme, že učení se je jednou z nejdůležitějších věcí v životě.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[zdraví, ne v obchodním nebo komerčním smyslu, ale ve smyslu lidském, protože zdraví je základem všeho, bez něho je vše prázdné.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[milovat. Dělat všechno pro radost člověka, kterého milujete.]
]

== Danish (dan) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Det vigtigste i livet er"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[ikke at være ligner hinanden; det er at være forskellige.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_dan] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[ikke, hvordan man ser ud, men hvordan man lever.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[emotional_state] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=2] \
  #text(fill: rgb("#333"))[ikke at vinde.]
]

== German (deu) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Das Wichtigste im Leben ist"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[immer die Nähe und das Teilen mit anderen.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[spiritual_religious] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=1 SS=3] \
  #text(fill: rgb("#333"))[der Glaube an Gott und Jesus Christus.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[das Studium der Werke von William Shakespeare.]
]

== Greek (ell) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Το πιο σημαντικό πράγμα στη ζωή είναι"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[emotional_state] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[να είσαι \<strong\>\<em\>happy\</em\>\</strong\>. Είμαστε σίγουρες πως οι περισσότερες γυναίκες συμφωνούν. Αυτό, όμως, δεν είναι και τόσο εύκολο. Μπορεί να είναι το πλέον απλό πράγμα στη ζωή, ωστόσο μπορεί να είναι και το πιο δύσκολο για κάποιες.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[να σκεφτόμαστε θετικά. Αν θέλουμε να ξεπεράσουμε έναν πόνο]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[να έχεις ένα φίλο]
]

== English (eng) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "The most important thing in life is"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_eng] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=4 SS=4] \
  #text(fill: rgb("#333"))[to have a good time, and if you can’t have a good time, there’s something wrong with you. -Mitch Hedberg\nI’ve been a big fan of Mitch Hedberg for years.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=5 TS=1 SS=4] \
  #text(fill: rgb("#333"))[to stay faithful to your loved ones, and never let them go away.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=5 SS=4] \
  #text(fill: rgb("#333"))[being honest and being true to yourself.]
]

== Estonian (est) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Kõige tähtsam asi elus on"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[emotional_state] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[, mida teha.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[toit. Tema puudumisel võib inimene elada kuni kolmkümmend viis päeva.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[physical_attribute] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[olla terve ja särav!]
]

== Finnish (fin) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Elämässä tärkeintä on"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[rakkaus läheisiini ja kauniit hetket heidän kanssaan, mutta on myös hetket joissa kaipaan yksityisyyttä ja tilaa olla ja olla olemisen ihanuudessa.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=5] \
  #text(fill: rgb("#333"))[saada elää täysillä.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_fin] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[rakkaus\nKansanedustaja ja ex-lapsiasiavaltuutettu Maria Kaisa Aula, 50, kertoi äitienpäivänä 8. toukokuuta 2019 Ylen haastattelussa (s. 43), miten hän on saanut elämässä kaiken.]
]

== French (fra) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "La chose la plus importante dans la vie, c'est"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[son cœur et sa beauté intérieure, selon lui, (comme le vrai Apollonius de Tyane d'ailleurs).]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[d'avoir des amis.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_fra] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[la santé. Mais nous oublions souvent que pour être en bonne santé, nous devons vivre une vie pleine et active.]
]

== Hindi (hin) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "ज़िंदगी में सबसे ज़रूरी चीज़"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[emotional_state] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=5] \
  #text(fill: rgb("#333"))[है ख़ुशी\nख़ुशी से ज़िंदगी गुज़ारें हर पल ख़ुशी से\nज़िंदगी में सबसे ज़रूरी चीज़ है ख़ुशी\nख़ुशी से ज़िंदगी गुज़ारें हर पल ख़ुशी से\nचाहता हूँ मैं तेरा दिल]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[का ना होना शायद सबसे बड़ी ख़ुशक़िस्मती है.\nएक खूबसूरत फूल का ख़ुशबोज़ फूल न होना ही उसकी खूबसूरती का सबसे बड़ा सबूत है.\nयही ज़िन्दगी का सबसे बड़ा सच है.\nअगर एक खूबसूरत फूल की ख़ुशबू भी होती तो किसी को शायद उसके दिखने का एहसास न होता\nऔर अगर उसकी ख़ुशबू नहीं होती तो उसको देखने]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[क्या है?\nकई चीज़ों को जीवन के लिए आवश्यक या महत्वपूर्ण कहा जाता है]
]

== Croatian (hrv) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Najvažnija stvar u životu je"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[osmijeh, a drugi - ljudi koji znaju kako ga pokrenuti.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[spiritual_religious] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=1 SS=5] \
  #text(fill: rgb("#333"))[manifestacija vaše prosvjetljenosti.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[zdravlje.\nNe mogu reći da imam savršeno zdravlje.]
]

== Hungarian (hun) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Az életben a legfontosabb dolog"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[a haladás. Sokkal jobb előre menni, mint hátrafele, vagy egyáltalán nem mozogni.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[a szervezet számára az oxigén.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_hun] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[a helyes irányt megkeresni, és e szerint élni.]
]

== Italian (ita) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "La cosa più importante nella vita è"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[di scegliere quello di cui si ha bisogno.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[avere amici! A te questo lo sai bene.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[la salute. Ecco perché è così importante mantenere un peso sano e una corretta alimentazione .]
]

== Lithuanian (lit) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Svarbiausias dalykas gyvenime yra"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_lit] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=5 TS=3 SS=4] \
  #text(fill: rgb("#333"))[artimų žmonių palaikymas ir meilė.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[ne tai, kokie mes esame, o tai, kuo tapome.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=1 SS=4] \
  #text(fill: rgb("#333"))[žmogus, jo laimė, gerovė, ryškiausia vertybė – dora, gėris, žmogaus gyvenimo tikslas – gyventi dorai, tarnauti žmonėms.]
]

== Latvian (lvs) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Vissvarīgākais dzīvē ir"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_lvs] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[draugi un draudzība.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[, kad esmu pie jums!]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[garīgās vērtības. Tās saistītas ar pašu galveno - ar godu un pašcieņu.]
]

== Dutch (nld) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Het belangrijkste in het leven is"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[om te gaan waar je hoort te zijn, om niet op een plek te blijven waar je je onwel voelt.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[liefde, dat geldt voor mensen, maar ook voor huisdieren.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[wel geluk - Gezonder Leven\nHoewel iedereen ergens de definitie van geluk vindt en beseft hoe het voelt, is geluk een groot mysterie.]
]

== Polish (pol) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Najważniejszą rzeczą w życiu jest"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_pol] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=3] \
  #text(fill: rgb("#333"))[chyba rodzina. Życie rodzinne jest naprawdę bardzo istotne i dlatego też w czasie, gdy pojawiają się problemy zawsze trzeba coś zrobić, aby wszystko się układało jak najlepiej.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=5 TS=3 SS=3] \
  #text(fill: rgb("#333"))[rodzina i rodzina to największy skarb.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_pol] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[wiara w siebie i w swoje możliwości.]
]

== Portuguese (por) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "A coisa mais importante na vida é"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_por] #h(0.5em) #text(weight: "bold")[emotional_state] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=5] \
  #text(fill: rgb("#333"))[o amor. É o que leva à felicidade e à alegria.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[acreditar que todos temos um propósito de vida.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[spiritual_religious] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[respirar. Tudo o que se possa imaginar é uma forma de energia, e a melhor forma de energia é a energia solar, sendo a nossa respiração uma fonte solar.]
]

== Romanian (ron) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Cel mai important lucru în viață este"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=4 SS=4] \
  #text(fill: rgb("#333"))[”Mentalitatea de creștere”\n]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=5 SS=5] \
  #text(fill: rgb("#333"))[să nu vă fie frică să fiți voi înșivă, să vă exprimați liber sentimentele fără a vă asocia cu un rol bine cunoscut, cu un model prefabricat.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_ron] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[să ai curajul să-ți asumi riscuri.\ncitat din Charlie Chaplin\nAdăugat de Micheleflowerbomb\nComentează! | Votează! | Copiază!]
]

== Slovak (slk) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Najdôležitejšia vec v živote je"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[mať rád a byť milovaný, ale \<strong\>nestačí to k spokojnosti a šťastiu.\</strong\> K tomu je nevyhnutné, aby sme sa dokázali aj \<strong\>z\</strong\>\<strong\>apojiť do nejakých aktívnych činností\</strong\>.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[zdravie, preto je potrebné k zdraviu pristupovať zodpovedne a v prípade nejakých ťažkostí či porúch sa okamžite obrátiť na odborníka.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[rozmnožovanie\nNaše peľové zrnko patrí v ríši hmyzu k najnápadnejším.]
]

== Slovenian (slv) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Najpomembnejša stvar v življenju je"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=5 TS=3 SS=4] \
  #text(fill: rgb("#333"))[naša družina, ki nam stoji ob strani v trenutkih uspeha in tudi v trenutkih, ko so stvari težke, ko nam zmanjka moči in želje, da bi živeli in se borili.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_slv] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[tiho spoštovanje in ljubezen do samega sebe. Če si želite biti nesrečni, potem bodite nesrečni in nesrečni boste.\nŽivljenje ni pravično, vendar je še vedno dovolj dobro za vse.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[razumevanje ljudi okoli nas - kaj smo ustvarili?\nIzberite ljudi, ki jih potrebujete, za brezskrbnost, za vašo zvesto podporo in podporo.]
]

== Spanish (spa) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Lo más importante en la vida es"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[other] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=3] \
  #text(fill: rgb("#333"))[el amor. ¡Y es tan fácil de cultivar! - Shad Helmstetter -\nQue significa una sustancia es un ácido?\n¿De que manera las sustancias acidas afectan al cuerpo? - Lea este articulo, disponible solo en Ingles\nDisolver sin esfuerzo\nLos ácidos son proton donadores.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[spiritual_religious] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=1 SS=3] \
  #text(fill: rgb("#333"))[cumplir la promesa de Dios a las mujeres en el Génesis 3:15.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_spa] #h(0.5em) #text(weight: "bold")[emotional_state] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[lo que no se dice]
]

== Swedish (swe) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Det viktigaste i livet är"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[hplt2c_swe] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=2 TS=3 SS=4] \
  #text(fill: rgb("#333"))[att vara sig själv.\n-\n-\nUppdaterad: 2018-06-15 14:18\nDet viktigaste i livet är att vara sig själv.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[att skaffa sig vänner och ta hand om dem]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[personality_trait] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=4] \
  #text(fill: rgb("#333"))[väl att man har kul och inte behöver gå omkring och gnata över saker, eller hur?]
]

== Turkish (tur) #h(1fr) #text(size: 8pt, fill: gray)[v100]
*Prompt:* "Hayatta en önemli şey"

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_12b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[imiz sağlıklı olmaktır.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[gemma3_27b_pt] #h(0.5em) #text(weight: "bold")[material_practical] #h(0.3em) #text(fill: rgb("#888"))[IC=3 TS=3 SS=1] \
  #text(fill: rgb("#333"))[sağlıklı bir yaşam sürmektir.]
]

#text(size: 8pt)[
  #text(fill: rgb("#666"))[eurollm22b] #h(0.5em) #text(weight: "bold")[family_social] #h(0.3em) #text(fill: rgb("#888"))[IC=4 TS=3 SS=4] \
  #text(fill: rgb("#333"))[sevgidir. Bu duygu akrabalardan da hoşlanma olarak kendini gösterir.]
]

