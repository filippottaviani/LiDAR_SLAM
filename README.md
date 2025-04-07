# LiDAR SLAM

Questo repository contiene il progetto relativo all'esame di Autonomous Robotics, sviluppato da Filippo Ottaviani e Samuele Ceccarelli.

## Descrizione

Il progetto implementa un sistema di Simultaneous Localization and Mapping (SLAM) utilizzando dati LiDAR attraverso la simulazione in Simulink di un veicolo che si muove in una scena 3D. In particolare si vuole permettere all'utente di scegliere una traiettoria sulla mappa, simulare un veicolo che la segue ed estrarre le misure dal sensore LiDAR posizionato sopra di esso. Una volta ottenuti ed elaborati opportunamente i dati, si desidera implementare un algoritmo di SLAM che permetta di ottenere una ricostruzione della mappa dell'ambiente e una stima della traiettoria. Queste verranno poi confrontate con il riferimento reale (ground truth) per verificare la bontà dell'algoritmo. 

## Struttura del Repository

- `animations/`: Contiene animazioni e visualizzazioni generate durante l'elaborazione dei dati.
- `images/`: Contiene le immagini risultanti dalle varie sezioni di codice.
- `functions/`: Include funzioni MATLAB personalizzate utilizzate nel progetto.
- `CustomLidarSLAM.slx`: Modello Simulink principale per l'implementazione del sistema SLAM.
- `lidar_slam_project.mlx`: Script MATLAB Live che documenta e guida attraverso le varie fasi del progetto.
- `lidar_slam_script.m`: Script MATLAB che contiene le varie fasi del progetto.

## Script MATLAB

### 1. **Tracciamento della traiettoria**
- Definizione della scena di simulazione.
- Selezione manuale dei waypoints.
- Generazione di un percorso continuo con interpolazione spline.

### 2. **Simulazione**
- Apertura del modello Simulink per la simulazione.
- Configurazione dei parametri di simulazione.
- Generazione delle pose di riferimento per la navigazione.

### 3. **Estrazione dei dati LiDAR**
- Acquisizione delle scansioni LiDAR durante la simulazione.
- Conversione delle scansioni in nuvole di punti.
- Visualizzazione dei dati in tempo reale.

### 4. **SLAM con LiDAR**
- Registrazione delle nuvole di punti consecutive tramite algoritmi LOAM (LiDAR Odometry and Mapping).
- Filtraggio dei dati per mantenere solo i punti rilevanti.
- Ottimizzazione dei parametri per l'estrazione delle feature.

### 5. **Costruzione della traiettoria**
- Stima della traiettoria senza mappatura basata sulla sola odometria.
- Stima della traiettoria con mappatura mediante l'aggiornamento iterativo di una mappa 3D.

### 6. **Valutazioni finali**
- Confronto tra traiettoria stimata e ground truth.
- Visualizzazione della mappa finale costruita.
- Calcolo degli errori tramite RMSE per valutare la precisione.

## Modello Simulink

### **1. Input: Posizioni di Riferimento**
- **refPosesX**: Coordinate X del veicolo nel percorso.
- **refPosesY**: Coordinate Y del veicolo nel percorso.
- **refPosesT**: Angolo di orientazione (Yaw) del veicolo.

Questi input definiscono la traiettoria di riferimento che il veicolo seguirà nella simulazione.

### **2. Simulazione del Veicolo 3D**
- Il blocco **"Simulation 3D Vehicle with Ground Following"** rappresenta un veicolo simulato in un ambiente 3D, in grado di seguire il terreno.
- Riceve in ingresso le coordinate \(X, Y\) e l'orientamento \(Yaw\).
- Il veicolo segue il percorso definito dagli input e genera dati di navigazione.

### **3. Sensore Lidar**
- Il sistema è dotato di un **sensore Lidar**, che acquisisce dati sull'ambiente circostante.
- Il blocco del sensore genera:
  - **Point Cloud (Nuvola di punti)**: Matrice \(32 X 2250 X 3\) contenente le misurazioni del Lidar.
  - **Translation (Traslazione)**: Vettore \(1 X 3\) con la posizione del Lidar.
  - **Rotation (Rotazione)**: Vettore \(1 X 3\) con l'orientamento del Lidar.

### **4. Output: Dati del Lidar**
I dati acquisiti dal sensore vengono inviati ai seguenti output:
- **out.lidarScans**: Nuvola di punti rilevati dal Lidar.
- **out.lidarLocation**: Posizione del sensore nel sistema di riferimento.
- **out.lidarOrientation**: Orientamento del sensore rispetto all'ambiente.

