from math import cos, pi, sin
import cv2
import glob
import numpy as np
import torch


#Code for the solution of the assigment

#Definicije funkcij

#Funkcija najde vse slike in resnice in vrne seznam poti do slik in resnic
def najdi_poti(pot):
    poti_do_slik = glob.glob(pot + '*.png')

    return poti_do_slik


def dodaj_resnico( pot_do_slike):
    slika_s_kvadratom = cv2.imread(pot_do_slike, 1)
    pot_do_resnice = pot_do_slike[:len(pot_do_slike)-3] + 'txt'

    with open (pot_do_resnice, 'rt') as resnica:
        for vrstica in resnica:
            _ , resnica_center_x , resnica_center_y ,resnica_širina , resnica_višina = [float(i) for i in vrstica.rstrip("\n").split(" ")]

    slika_h, slika_w, _ = slika_s_kvadratom.shape

    #pretvorba v pixel vrednosti širine in višine 
    resnica_širina = resnica_širina * slika_w
    resnica_višina = resnica_višina * slika_h

    #pretvorba v pixel vrednosti in postavitev na levi zgornji rob 
    resnica_center_x = resnica_center_x * slika_w - (resnica_širina/2)
    resnica_center_y = resnica_center_y * slika_h - (resnica_višina/2)

    #izris kvadrata 
    zacetek_kvadrata = (round(resnica_center_x), round(resnica_center_y))
    konec_kvadrata = (round(resnica_center_x + resnica_širina), round( resnica_center_y + resnica_višina))
    barva_kvadrata = (64,170,0)
    debelina_kvadrata = 5

    slika_s_kvadratom = cv2.rectangle(slika_s_kvadratom, zacetek_kvadrata, konec_kvadrata, barva_kvadrata, debelina_kvadrata)


    return slika_s_kvadratom


def dobi_resnico(pot_do_slike):
    slika = cv2.imread(pot_do_slike, 1)
    pot_do_resnice = pot_do_slike[:len(pot_do_slike)-3] + 'txt'

    with open (pot_do_resnice, 'rt') as resnica:
        for vrstica in resnica:
            _ , resnica_center_x , resnica_center_y ,resnica_širina , resnica_višina = [float(i) for i in vrstica.rstrip("\n").split(" ")]

    slika_h, slika_w, _ = slika.shape

    #pretvorba v pixel vrednosti širine in višine 
    resnica_širina = resnica_širina * slika_w
    resnica_višina = resnica_višina * slika_h

    #pretvorba v pixel vrednosti in postavitev na levi zgornji rob 
    resnica_center_x = resnica_center_x * slika_w - (resnica_širina/2)
    resnica_center_y = resnica_center_y * slika_h - (resnica_višina/2)

    #vrni obliko (x1,y1,x2,y2)
    return [int(round(resnica_center_x)), int(round(resnica_center_y)), int(round(resnica_center_x + resnica_širina)), int(round(resnica_center_y + resnica_višina))]


#izriše podano sliko
def izrisi_sliko(slika):

    cv2.imshow('IZRIS SLIKE', slika)

    #kle morš pol prtisnt neko tipko da se zapre
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    return 0

#funkcija na sliko doda skatle
#skatla mora biti oblike (x1,y1,x2,y2)
def dodaj_skatlo(slika, skatla, barva):
    if len(skatla) == 0:
        return 0
    for (x1,y1,x2,y2) in skatla:
        cv2.rectangle(slika, (x1, y1),(x2, y2), barva, 5)
    return 0

#funkcija izračuna VJ skatle s podanimi parametri in jih vrne v obliki [x,y,w,h], [x,y,w,h], ...
def izracunaj_Haar(pot_do_slike, scaleFaktor, minSosedov ,levo_uho_filter_VJ, desno_uho_filter_VJ):

    
    #odpri sliko v 
    slika_grey = cv2.imread(pot_do_slike, 0)



    #zaznaj ušesa
    levo_uho_skatla = levo_uho_filter_VJ.detectMultiScale(slika_grey, scaleFactor = scaleFaktor, minNeighbors = minSosedov)
    desno_uho_skatla = desno_uho_filter_VJ.detectMultiScale(slika_grey, scaleFactor = scaleFaktor, minNeighbors = minSosedov)

    #če najdeš obe ušesi vrni oboje kot en seznam
    if(len(levo_uho_skatla) > 0 and len(desno_uho_skatla) > 0):
        return np.concatenate([levo_uho_skatla,desno_uho_skatla])
    
    #če ne najdeš levih vrni samo desna
    if (len(levo_uho_skatla) < 1):
        return desno_uho_skatla

    #če ne najdeš desnih vrni samo leva
    if (len(desno_uho_skatla) < 1):
        return levo_uho_skatla

    #če ne najdeš nič vrni prazen seznam 
    return [[]]

#funkcija izracuna iou med dvema skatlama
def get_iou(skatla1, skatla2):
    """
    izračunaj IOU med škatlama 

    Vhod:
    skatla1 : seznam['x1', 'y1', 'x2', 'y2']
        (x1, y1) lev zgornji kot skatle 
        (x2, y2) desn spodnji kot skatle 

    skatla2 : seznam['x1', 'y1', 'x2', 'y2']
        (x1, y1) lev zgornji kot skatle 
        (x2, y2) desn spodnji kot skatle 

    Vrne IOU (float)
    """
    #koordinate območja prekrivanja 
    x_levo = max(skatla1[0], skatla2[0])
    y_zgoraj = max(skatla1[1], skatla2[1])
    x_desno = min(skatla1[2], skatla2[2])
    y_spodaj = min(skatla1[3], skatla2[3])

    #če se ne prekrivata vrni 0 
    if x_desno < x_levo or y_spodaj < y_zgoraj:
        return 0.0

    # površina območja prekrivanja
    obmocje_prekrivanja = (x_desno - x_levo) * (y_spodaj - y_zgoraj)

    #površina obeh škatel
    skatla1_povrsina = (skatla1[2] - skatla1[0]) * (skatla1[3] - skatla1[1])
    skatla2_povrsina = (skatla2[2] - skatla2[0]) * (skatla2[3] - skatla2[1])

    #iou škatel
    iou = obmocje_prekrivanja / float(skatla1_povrsina + skatla2_povrsina - obmocje_prekrivanja)

    #iou more bit med 0 in 1
    assert iou >= 0.0
    assert iou <= 1.0

    return iou

#funkcija pretvori obliko skatel iz [x,y,w,h] v [x1,y1,x2,y2]
def pretvor_obliko(seznam_skatel):
    seznam_novih_oblik = []
    #če je seznam prazen vrni prazen seznam 
    if len(seznam_skatel) == 0:
        #print("zaznal sem da je prazen seznam")
        return []

    for (x, y ,w ,h )in seznam_skatel:
            nova_oblika = []
            nova_oblika.append(x)
            nova_oblika.append(y)
            nova_oblika.append(x + w)
            nova_oblika.append(y + h)
            seznam_novih_oblik.append(nova_oblika)

    return seznam_novih_oblik

#funkcija izracuna yolo skatle in jih vrne v obliki [x1,y1,x2,y2], [x1,y1,x2,y2], ...
def yolo_model_skatle(pot_do_slike, model):
    

    rezultat_yolo = model(pot_do_slike)
    skatle_yolo = []
    for skatla in rezultat_yolo.xyxy[0].tolist():
        x1, y1, x2, y2, sigurnost, _ = [float(i) for i in skatla]
        skatla_yolo = [round(x1), round(y1), round(x2), round(y2)]
        skatle_yolo.append(skatla_yolo)

    return skatle_yolo

#funkcija pridobi tp fp in fn za dolocen threshold. Vrne tuple (tp, fp, fn)
def dobi_tpfpfn(resnica, seznam_skatel, threshold): 
    tp = 0
    fp = 0
    fn = 0
    
    # če nisi nič zaznal je to False Negative fn++
    if len(seznam_skatel) == 0:
        fn += 1
    #če si nekaj zaznal
    else:
        for skatla in seznam_skatel:
            #izracunaj iou skatle
            iou = get_iou(resnica,skatla)

            #če je iou 0 (ni prekrivanja) je to FP
            if iou == 0:
                fp += 1
            #če je prekrivanje 
            else: 
                #in je znotraj thresholda je to TP
                if iou >= threshold:
                    #tu moraš pazit da je lahko samo en tp na sliko
                    if tp >= 1:
                        #ostale šteješ za FP 
                        fp += 1 
                    else:
                        tp += 1 
                #in ni znotraj thresholda je to FP
                elif iou < threshold:
                    fp += 1 

    return tp, fp, fn

#funkcija obdela vse slike 
def obdelaj_slike(poti_do_slik, scaleFactor, stSosedov, threshold):
    #naloži filtre 
    levo_uho_filter_VJ = cv2.CascadeClassifier('Support Files\haarcascade_mcs_leftear.xml')
    desno_uho_filter_VJ = cv2.CascadeClassifier('Support Files\haarcascade_mcs_rightear.xml')

    model  = torch.hub.load( 'yolov5', 'custom', path='Support Files/yolo5s.pt', source="local")

    # za vse slike dub škatle in zračuni tp fp in fn 
    yolo_global_tpfpfn = (0, 0, 0)
    vj_global_tpfpfn = (0, 0, 0)

    for pot_do_slike in poti_do_slik:
        
        #pridobi resnico škatle
        resnica = dobi_resnico(pot_do_slike)

        #pridobi seznam škatel yolo. Oblika [[x1, y1, x2, y2],[skatla2], ...]
        skatle_yolo = yolo_model_skatle(pot_do_slike,model)

        #pridobi seznam škatel VJ. Oblika [[x,y,w,h]]
        skatle_VJ = izracunaj_Haar(pot_do_slike, scaleFactor, stSosedov, levo_uho_filter_VJ, desno_uho_filter_VJ)

        #pretvori VJ skatle v obliko [[x1, y1, x2, y2],[skatla2], ...]]
        skatle_VJ = pretvor_obliko(skatle_VJ)

        #pridobi tp, fp, fn za yolo 
        yolo_tpfpfn = dobi_tpfpfn(resnica,skatle_yolo,threshold)

        #pridobi tp, fp, fn za VJ
        vj_tpfpfn = dobi_tpfpfn(resnica,skatle_yolo,threshold)

        yolo_global_tpfpfn = tuple(map(sum, zip(yolo_tpfpfn, yolo_global_tpfpfn)))

        vj_global_tpfpfn = tuple(map(sum, zip(vj_tpfpfn, vj_global_tpfpfn)))

    return [yolo_global_tpfpfn, vj_global_tpfpfn]



# ------------------------------------------ Začetek "izvajanja" programa --------------------------------------------------------------------------
pot_do_testnih_podatkov = "Support Files/ear_data/test/"

poti_do_slik = glob.glob(pot_do_testnih_podatkov + '*.png')
#print(poti_do_slik[0], poti_do_resnic[0])


deset_slik = [poti_do_slik[0],poti_do_slik[1],poti_do_slik[2], poti_do_slik[3], poti_do_slik[4],poti_do_slik[5],poti_do_slik[6],poti_do_slik[7],poti_do_slik[8],poti_do_slik[9],poti_do_slik[10]]

rezultati_tpfpfn = obdelaj_slike(deset_slik, 1.03, 4, 0.5)

print(rezultati_tpfpfn)

yolo_iou = rezultati_tpfpfn[0][0] / (sum(rezultati_tpfpfn[0]))
print(yolo_iou)

vj_iou = rezultati_tpfpfn[1][0] / (sum(rezultati_tpfpfn[1]))
print(vj_iou)




