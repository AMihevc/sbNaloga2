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
    poti_do_resnic = glob.glob(pot + '*.txt')

    return poti_do_slik, poti_do_resnic


def dodaj_resnico(slika, pot_do_slike):

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

    #izris kvadrata 
    zacetek_kvadrata = (round(resnica_center_x), round(resnica_center_y))
    konec_kvadrata = (round(resnica_center_x + resnica_širina), round( resnica_center_y + resnica_višina))
    barva_kvadrata = (64,170,0)
    debelina_kvadrata = 5

    slika_s_kvadratom = cv2.rectangle(slika, zacetek_kvadrata, konec_kvadrata, barva_kvadrata, debelina_kvadrata)


    return slika_s_kvadratom


def izrisi_sliko(slika):

    cv2.imshow('IZRIS SLIKE', slika)

    #kle morš pol prtisnt neko tipko da se zapre
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    return 0


def yolo_model_skatle(pot_do_slike):
    model  = torch.hub.load( 'yolov5', 'custom', path='Support Files/yolo5s.pt', source="local")

    rezultat_yolo = model(pot_do_slike)

    return rezultat_yolo


#funkcija doda skatlo na sliko
#skatla mora biti oblike (x,y,w,h)
def dodaj_skatle(slika, skatla):
    for (x,y,w,h) in skatla:
        cv2.rectangle(slika, (x, y),(x + w, y + h), (0, 0, 255), 5)
    return 0


#vrne seznam vseh najdenih škatel z ušesi
def izracunaj_Haar(slika, scaleFaktor, minSosedov):

    #pretvori sliko v gray scale
    slika_grey = cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY)

    #naloži filtre 
    levo_uho_filter_VJ = cv2.CascadeClassifier('Support Files\haarcascade_mcs_leftear.xml')
    desno_uho_filter_VJ = cv2.CascadeClassifier('Support Files\haarcascade_mcs_rightear.xml')

    #zaznaj ušesa
    levo_uho_skatla = levo_uho_filter_VJ.detectMultiScale(slika_grey, scaleFactor = scaleFaktor, minNeighbors = minSosedov)
    desno_uho_skatla = desno_uho_filter_VJ.detectMultiScale(slika_grey, scaleFactor = scaleFaktor, minNeighbors = minSosedov)

    #če najdeš obe ušesi vrni oboje kot en seznam
    if(len(levo_uho_skatla) > 0 & len(desno_uho_skatla) > 0):
        return np.concatenate([levo_uho_skatla,desno_uho_skatla])
    
    #če ne najdeš levih vrni samo desna
    if (len(levo_uho_skatla) < 1):
        return desno_uho_skatla

    #če ne najdeš desnih vrni samo leva
    if (len(desno_uho_skatla) < 1):
        return levo_uho_skatla

    #če ne najdeš nič vrni prazen seznam 
    return [[]]


def pretvor_obliko (seznam_skatel):
    seznam_novih_oblik = []

    for (x, y ,w ,h )in seznam_skatel:
            nova_oblika = []
            nova_oblika.append(x)
            nova_oblika.append(y)
            nova_oblika.append(x + w)
            nova_oblika.append(y + h)
            seznam_novih_oblik.append(nova_oblika)

    return seznam_novih_oblik


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
    # determine the coordinates of the intersection rectangle
    x_levo = max(skatla1[0], skatla2[0])
    y_zgoraj = max(skatla1[1], skatla2[1])
    x_desno = min(skatla1[2], skatla2[2])
    y_spodaj = min(skatla1[3], skatla2[3])

    if x_desno < x_levo or y_spodaj < y_zgoraj:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    obmocje_prekrivanja = (x_desno - x_levo) * (y_spodaj - y_zgoraj)

    # compute the area of both AABBs
    skatla1_povrsina = (skatla1[2] - skatla1[0]) * (skatla1[3] - skatla1[1])
    skatla2_povrsina = (skatla2[2] - skatla2[0]) * (skatla2[3] - skatla2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = obmocje_prekrivanja / float(skatla1_povrsina + skatla2_povrsina - obmocje_prekrivanja)
    assert iou >= 0.0
    assert iou <= 1.0
    
    return iou

# ------------------------------------------ Začetek "izvajanja" programa --------------------------------------------------------------------------
pot_do_testnih_podatkov = "Support Files/ear_data/test/"

poti_do_slik, poti_do_resnic = najdi_poti(pot_do_testnih_podatkov)
#print(poti_do_slik[0], poti_do_resnic[0])



