# Nedeljni izveštaj

> **NACRT — potrebno je da učenik potvrdi tačnost pre finalizacije.**

## 1. Osnovne informacije

| Polje | Odgovor |
| --- | --- |
| Ime i prezime | Matija Radulovic |
| Adresa e-pošte | matija.radulovic077@gmail.com |
| Discord korisničko ime | Matija Radulovic |
| Nedelja | Nedelja 6 — RAG |
| Par / tim | Uros; zadatak smo radili svako za sebe |
| Moj konkretan doprinos / uloga | Samostalno sam proširio korpus, unapredio similarity scoring za retrieval, napravio i analizirao eksperimente za chunk size i overlap, izabrao podrazumevana podešavanja i proverio citiranje, odbijanje nepodržanih pitanja i završne dokaze. |
| Datum predaje | 2026-10-14 |
| Reference na rad i dokaze | Commit-i `813d768`, `f426961`, `ed539da`, `f524653` i `604fa5a`; `evidence.md`; `docs/work_log.md`; komande `python3 -m scripts.analyze_chunking`, `python3 -m scripts.check_retrieval`, `python3 -m scripts.check_citations`, `python3 -m scripts.check_refusal`, `python3 -m scripts.generate_evidence` i `./scripts/run_tests.sh`. |

## 2. Moj status

**Status:** Završeno

Završio sam svih pet ishoda iz domaćeg zadatka. Napravio sam ponovljive provere za chunking, top-k retrieval, citate, refusal ponašanje i završni par podržanog i nepodržanog pitanja. Završna provera je imala rezultat `15 passed`, a svi posebni evidence skriptovi završili su se sa `PASS`. Dokumentovani su korišćene komande, rezultati i obrazloženja. Jedina administrativna stavka koja nije deo tehničke implementacije jeste određivanje demo lead-a.

## 3. Rad ove nedelje

Ove nedelje radio sam na RAG zadatku koji pokriva pripremu korpusa, chunking sa overlap-om, top-k retrieval, odgovore sa citatima i odbijanje pitanja kada korpus nema dovoljno dokaza. Uros i ja smo bili par, ali smo zadatak radili svako za sebe. Moj rad i dokazi nalaze se u lokalnom Git repozitorijumu i u fajlovima `evidence.md` i `docs/work_log.md`.

Prvo sam napravio skriptu koja poredi više konfiguracija za veličinu chunk-a i overlap. Skripta koristi tri karakteristična pitanja sa unapred poznatim relevantnim dokumentima i ispisuje rangirane chunk-ove, hit rate, mean reciprocal rank i prosečan najbolji skor. Poredio sam konfiguracije od `80/0` do `220/50`. Sve su postigle `1.000` MRR, ali je posle proširenja korpusa i izmene scoring-a konfiguracija `100/20` imala najbolji prosečan top score od `0.465`. Na osnovu tih rezultata izabrao sam `chunk_size=100` i `overlap=20` kao podrazumevane vrednosti. Ovo podešavanje čuva deo konteksta na granici chunk-ova i dalo je bolje fokusirane rezultate od širih chunk-ova.

Zatim sam proširio početni korpus na pet dužih dokumenata. Dodao sam sadržaj o evaluaciji retrieval-a i kvalitetu citata, a postojeće dokumente o RAG-u, chunking-u i refusal ponašanju sam produžio. Time je eksperiment postao zahtevniji jer retrieval sada bira između više tematski sličnih delova. Unapredio sam i similarity scoring: uklanjaju se česte stop reči, porede se jedinstveni značajni termini pomoću binarne cosine sličnosti, a naslov dokumenta daje manji dodatni signal. Rezultati se sortiraju opadajuće po skoru, uz determinističko razrešavanje jednakih rezultata. Provera za `top_k=2` i `top_k=3` pokazala je da su prva dva rezultata iz dokumenta o chunking-u i da se poštuje traženi broj rezultata.

Promene su organizovane kroz sledeće commit-e:

- `813d768` je dodao dokumentaciju, detaljan plan rada i početne dokaze za prva dva ishoda.
- `f426961` je implementirao eksperimente za chunking, duži korpus, novi top-k retrieval scoring, testove i skripte za proveru Ishoda 1 i 2.
- `ed539da` je završio Ishod 3. Odgovor sada ima odvojene sekcije `Answer` i `References`, a svaki podržavajući deo sadrži tačan chunk ID. Provera potvrđuje da se citati tačno poklapaju sa vraćenim chunk-ovima.
- `f524653` je završio Ishod 4. Uvedeno je pravilo dovoljne evidencije kako jedno slučajno poklapanje reči ne bi proizvelo nepodržan odgovor. Pitanje o boji retrieval-a na Marsu vratilo je slabe rezultate, ali je sistem ispravno odbio odgovor bez citata.
- `604fa5a` je završio Ishod 5 i dodao jednu komandu koja prikazuje kompletan podržani i nepodržani primer, uključujući pitanje, retrieved chunk-ove, skorove, odgovor, citate i refusal status.

AI sam koristio za pomoć pri kodiranju i organizaciji dokumentacije. Pregledao sam sve predložene promene i sve izlaze komandi. Lično sam proverio rezultate eksperimenta, rangiranje, citate, refusal izlaz, završni evidence prikaz i testove. Prihvatio sam promene tek nakon tih provera; sve promene su mi jasne i završni rezultat izgleda ispravno. AI izlaze nisam tretirao kao dokaz bez pokretanja odgovarajućih skripti i testova.

Važan zaključak mi je da izbor chunk size-a ne treba donositi samo na osnovu jednog similarity skora. Kratki chunk-ovi mogu veštački izgledati bolji, dok preveliki chunk-ovi razvodnjavaju relevantne termine. Zato sam odluku zasnovao na rangiranju kroz više pitanja, overlap-u koji čuva kontekst i poređenju prosečnih rezultata. Trenutno mi nije potrebna pomoć tutora.

## 4. Sledeći korak

Sledeći korak mi je da implementiram chunking prema sadržaju i prirodnim granicama teksta, umesto samo prema broju karaktera, i da ga proverim na složenijem korpusu napravljenom od jedne knjige. Korak će biti završen kada uporedim novi pristup sa sadašnjim podešavanjem `100/20` na istim metrikama i zabeležim rezultate za najmanje tri karakteristična pitanja.

## 5. Poverljiva napomena za tutora

Nema dodatne napomene.
