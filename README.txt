1. simulation_7030.py
=====================

Simulazione del modello di Wilson-Cowan stocastico (popolazioni
eccitatorie/inibitorie, con chiE = 70%, chiI = 30%) su una griglia
di parametri di disaccoppiamento sinaptico delta_EI x delta_IE.

Le formule analitiche usate in compute_analytics (coefficienti
della matrice di drift, autovalori, matrice di covarianza
stazionaria) sono prese da:

  M. K. Nandi, A. de Candia, A. Sarracino, H. J. Herrmann,
  L. de Arcangelis, "Fluctuation-dissipation relations in the
  imbalanced Wilson-Cowan model", Phys. Rev. E 107, 064307 (2023).
  DOI: 10.1103/PhysRevE.107.064307

in particolare le Eq. (A16)-(A21) dell'Appendice (coefficienti di
A_EE, A_EI, A_IE, A_II e ampiezze di rumore) e le Eq. (A32)-(A34)
(matrice di covarianza sigma11, sigma12, sigma22).

Per ogni punto della griglia:

  - integra numericamente (Numba, parallelizzato con prange) le
    equazioni di Langevin non lineari per il numero di neuroni
    attivi eccitatori (k) e inibitori (l), con passo temporale
    dt = 1e-3 ms e T = 32000 ms (di cui la seconda meta' usata per
    le statistiche stazionarie);

  - calcola le medie stazionarie E0, I0 e le fluttuazioni xiE, xiI;

  - calcola analiticamente (linearizzazione del modello attorno al
    punto fisso) i coefficienti della matrice di drift A_EE, A_EI,
    A_IE, A_II, gli autovalori (reali/complessi) lambda1, lambda2,
    lambda_re, lambda_im, i coefficienti della matrice trasformata
    (variabili Sigma, Delta) A_x, A_y, A_z, A_w, la matrice di
    covarianza stazionaria sigma11, sigma12, sigma22 e
    l'attivita'/imbalance al punto fisso Sigma0, Delta0.

I risultati per ogni valore del campo esterno h (in [1e-9, 1e-10]
nello script attuale) vengono salvati in
grid_dynamics_7030_h{h}.npz e .mat.

NOTA: la formula usata per sigma12 in compute_analytics contiene un
errore di segno nel termine con H rispetto all'Eq. (A33)
dell'articolo (vedi sotto, correzione.py).


2. correzione.py
================

Script di post-processing che ricalcola sigma11, sigma12, sigma22
a partire dai .npz gia' prodotti da simulation_7030.py, correggendo
un errore di segno nella formula di sigma12 (Eq. A33 dell'articolo
di Nandi et al., PRE 107, 064307 (2023)), derivata dalla soluzione
stazionaria dell'equazione di Lyapunov A*sigma + sigma*A^T = 2M:

  - formula usata in simulation_7030.py (errata):
      sigma12 = -(G*(z*w + x*y) + 2*H*x*w) / denom

  - formula corretta (Eq. A33):
      sigma12 = -(G*(z*w + x*y) - 2*H*x*w) / denom

Lo script legge ogni file grid_dynamics_7030_h{h}.npz per una
lista di valori di h, ricalcola le tre matrici di covarianza con
la formula corretta, e salva una copia con suffisso
_sigmaFIX.npz, lasciando invariati tutti gli altri campi salvati
dalla simulazione originale.

Correzione equivalente "a monte": nella riga di
simulation_7030.py che calcola sigma12 dentro compute_analytics,
basta cambiare il segno del termine 2.0 * H * x * w da + a - per
ottenere direttamente i dati corretti senza bisogno del
post-processing.


3. FDR_integrale.ipynb
============================
Violazione della fluttuazione-dissipazione (I_t) al variare di h

Il notebook analizza, per il modello di Wilson-Cowan sbilanciato
(Nandi et al., Phys. Rev. E 107, 064307, 2023), quanto il sistema
si discosti dalla relazione di fluttuazione-dissipazione (FDT) al
variare del campo esterno h (10^-6, 10^-7, 10^-8, 10^-9), sulla
griglia dei parametri (delta_EI, delta_IE).

I dati (E0, I0, elementi della Jacobiana, matrice ruotata,
covarianza, autovalori) sono caricati da file .npz gia' prodotti
dalla simulazione: nel notebook non viene ricalcolata alcuna
quantita' di base, solo quantita' derivate (tasso di entropia e
distanza dalla FDT).

Definite le quantita' normalizzate:

  x(t) = C_SigmaSigma(t) / C_SigmaSigma(0)
  y(t) = chi_SigmaSigma(t) / C_SigmaSigma(0)

la FDT prevede y(t)/y_inf = 1 - x(t); la distanza da questa
relazione e' quantificata da:

  I_t = (1/T) * integrale da 0 a T di
        |y(t)/y_inf - (1 - x(t))| dt

calcolata analiticamente lungo linee a delta_EI fissato e a
delta_IE fissato, per tutti e 4 gli h.

SCELTA DEL TEMPO DI INTEGRAZIONE T

Non e' un valore arbitrario ne' diverso per ogni h: viene
calcolato una sola volta e riusato ovunque, cosi' che I_t resti
confrontabile tra h diversi. Il procedimento:

  - per ogni punto delle linee usate si stima il tempo di
    rilassamento tau_slow = -1/lambda dall'autovalore meno
    negativo (parte reale, se complesso);

  - si prende il tau_slow massimo su tutte le linee e su tutti
    gli h (il caso piu' lento in assoluto, tipicamente h=10^-9,
    piu' vicino alla criticita');

  - si moltiplica per un margine K=21 (scelto in base alla
    precisione dei dati, arrotondati a 9 cifre decimali);

  - il risultato viene infine limitato tra un pavimento (1000) e
    un tetto di sicurezza (250000, per non far esplodere il tempo
    di calcolo vicino alla criticita' pura).

Su questo T viene costruita una griglia temporale comune di
200.000 punti, usata identica per tutte le linee e tutti gli h.

DIAGNOSTICA DEI PICCHI

Il notebook individua e tabella i massimi di I_t lungo ciascuna
linea, e verifica - sia graficamente sia con un controllo
numerico esplicito - che questi picchi coincidano con gli zeri
dell'elemento w della matrice di drift A (in variabili Sigma,
Delta): dove w=0 la normalizzazione di I_t per y_inf diventa
singolare, generando un picco per costruzione, indipendente da h
lontano dalla criticita'.

Infine confronta l'altezza dei picchi (I_max) e il tempo di
rilassamento associato tra i 4 valori di h, per stabilire se e
quanto la violazione della FDT dipenda dall'intensita' del campo
esterno.


4. beta_eff.ipynb
=====================
Analisi di beta_eff (temperatura effettiva) al variare di h

Il notebook analizza, per il modello di Wilson-Cowan sbilanciato
(Nandi et al., Phys. Rev. E 107, 064307, 2023), la "temperatura
effettiva" beta_eff del sistema al variare del campo esterno h
(10^-6, 10^-7, 10^-8, 10^-9), sulla griglia dei parametri
(delta_EI, delta_IE).

I dati (E0, I0, elementi della Jacobiana, matrice ruotata,
covarianza, autovalori) sono caricati da file .npz gia' prodotti
dalla simulazione: nel notebook non viene ricalcolata alcuna
quantita' di base, solo quantita' derivate (tasso di entropia e
beta_eff).

CALCOLO DI BETA_EFF

beta_eff e' definito come y(infinito), il valore asintotico della
risposta integrata chi_SigmaSigma(t)/C_SigmaSigma(0). Poiche'
nella regione studiata tutti gli autovalori del sistema sono
negativi (il sistema e' sempre stabile), il limite t -> infinito
ha una forma chiusa analitica: non serve integrare nel tempo,
beta_eff si calcola direttamente dai coefficienti gia' salvati
(autovalori, sigma11, elemento x della matrice ruotata),
separatamente per il caso a autovalori reali e per quello a
autovalori complessi.

GRAFICI

  - Scatter separati beta_eff>0 e beta_eff<0: due pannelli
    distinti, uno per segno. Nel primo sono colorati solo i punti
    con beta_eff positivo, nel secondo solo quelli con beta_eff
    negativo; in entrambi i pannelli i punti dell'altro segno
    sono mostrati in nero, cosi' da vedere subito dove il segno
    cambia sul piano (delta_IE, delta_EI). La scala colore e'
    automatica per ciascun pannello h.

  - Scatter su scala fissa [-20, 20]: stessa griglia, ma senza
    separare i segni e senza mascherare nulla; tutti i punti sono
    colorati su un'unica scala lineare comune a tutti gli h, da
    -20 a 20. I punti molto vicini alla singolarita' (dove il
    denominatore della formula di beta_eff si annulla, w quasi
    zero) possono avere valori reali molto piu' estremi
    dell'intervallo: in quel caso appaiono semplicemente
    "clippati" al colore limite.

VERIFICA: FORMULA CHIUSA VS STIMA TAIL-MEAN

In altre parti dell'analisi (calcolo di I_t) y_infinito viene
stimato numericamente come media di y(t) sull'ultimo 10% di un
intervallo temporale comune (tail-mean), anziche' con la formula
chiusa. Questa sezione confronta le due stime punto per punto
sulla griglia, per le stesse linee a delta_EI/delta_IE fissato
usate altrove e per tutti gli h, per verificare che coincidano
entro una soglia di scarto relativo (0.1%).

Per fare questo confronto e' necessario ricostruire lo stesso
intervallo temporale comune (t_common) usato altrove: la sua
costruzione (scelta del tempo di integrazione a partire dal tempo
di rilassamento piu' lento del sistema) e' inclusa in questa
sezione solo come prerequisito tecnico per il confronto, non fa
parte dell'analisi di beta_eff in se'.

Il risultato finale e' una tabella con il numero di punti che
divergono oltre soglia e lo scarto massimo osservato, per
ciascun h.

5. Wilson_Cowan.ipynb
=====================
Esplorazione completa del modello sbilanciato, per i 4 h

Notebook riassuntivo che raccoglie in un unico file, per il
modello di Wilson-Cowan sbilanciato (Nandi et al., Phys. Rev. E
107, 064307, 2023) e per tutti e 4 i valori di h (10^-6, 10^-7,
10^-8, 10^-9), i principali grafici gia' sviluppati nei notebook
precedenti (confronto_h_PULITO.ipynb e wilson_cowan_explorer.ipynb),
in un'unica sequenza.

I dati sono caricati direttamente dai file .npz gia' prodotti
dalla simulazione (tutti i campi disponibili, non solo un
sottoinsieme): nel notebook non viene ricalcolata alcuna
quantita' di base, solo quantita' derivate (log10, tasso di
entropia, beta_eff).

CONTENUTO, IN ORDINE

  1) E0, I0, Sigma0, Delta0

     Per ciascuna delle quattro quantita', due grafici: uno in
     scala lineare e uno in scala logaritmica (log10). Per i
     grafici logaritmici i valori minori o uguali a zero vengono
     mascherati (messi a NaN), perche' il logaritmo non e'
     definito li' - in particolare Delta0 puo' essere negativo
     (sbilanciamento a favore dell'inibizione).

  2) Correlazioni e risposte

     Le 4 funzioni di correlazione (C_SigmaSigma, C_SigmaDelta,
     C_DeltaSigma, C_DeltaDelta) e le 4 funzioni di risposta
     (R_SigmaSigma, R_SigmaDelta, R_DeltaSigma, R_DeltaDelta),
     calcolate analiticamente (caso reale e caso complesso) in un
     punto della griglia scelto manualmente (delta_IE_target,
     delta_EI_target). Il grafico mostra una riga per ciascun h:
     a sinistra le correlazioni, a destra le risposte.

  3) Diagramma degli autovalori (ricostruzione Fig. 2 di Nandi et
     al.), uno per ciascuno dei 4 h

     Quattro pannelli per ogni h: regione a autovalori reali o
     complessi, parte immaginaria (solo dove complessi), parte
     reale di lambda1 e parte reale di lambda2.

  4) Confronto tra gli autovalori a h=10^-6 e h=10^-9

     Stessa struttura a 4 pannelli del punto precedente, ma con
     la differenza punto per punto tra i due h piu' estremi:
     differenza di regime, differenza di parte immaginaria (solo
     dove entrambi complessi), differenza di Re(lambda1) e di
     Re(lambda2).

  5) Tasso di produzione di entropia

     - uno scatter/contour per ciascuno dei 4 h (in scala log10,
       perche' il tasso di entropia puo' variare su molti ordini
       di grandezza ed e' sempre non negativo);
     - il tasso di entropia lungo linee a delta_EI fissato (con
       delta_IE variabile);
     - il tasso di entropia lungo linee a delta_IE fissato (con
       delta_EI variabile).

  6) beta_eff (temperatura effettiva)

     - scatter separati per segno: un pannello con solo i punti
       beta_eff>0 (il resto in nero) e uno con solo i punti
       beta_eff<0 (il resto in nero), scala colore automatica per
       ciascun pannello h;
     - scatter su scala lineare fissa [-20, 20], senza separare i
       segni e senza mascherare nulla (tutti i punti sulla stessa
       scala per tutti gli h).

  7) Diagnostica del ramo complesso di beta_eff

     Scompone la formula beta_eff = (X - 2a) / (sigma11*(a^2+b^2))
     nei suoi pezzi (parte reale e immaginaria dell'autovalore,
     sigma11, numeratore, denominatore, beta_eff), solo nella
     regione a autovalori complessi, per capire dove e perche' il
     denominatore collassa. Di default e' mostrata per h=10^-6;
     basta cambiare la variabile h_label nella cella per ripetere
     la diagnostica su un altro h.