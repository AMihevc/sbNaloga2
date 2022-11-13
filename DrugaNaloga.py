from math import cos, pi, sin
import cv2
import glob
import numpy as np
import torch


#Code for the solution of the assigment
'''TODO
implment yolo5
    - odpreš vse slike z glob
    - za vsako slikoo runaš model iz yolo 
    - primerjaš z ground truth, ki je v .txt filih za vsako sliko

implement Haar
    - odpreš vsako lsiko z glob
'''

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

def yolo_model(slika):
    model  = torch.hub.load( 'yolov5', 'custom', path='Support Files/yolo5s.pt', source="local")

    rezultat_yolo = model(slika)

    return rezultat_yolo


# ------------------------------------------ Začetek "izvajanja" programa --------------------------------------------------------------------------
pot_do_testnih_podatkov = "Support Files/ear_data/test/"

poti_do_slik, poti_do_resnic = najdi_poti(pot_do_testnih_podatkov)
#print(poti_do_slik[0], poti_do_resnic[0])


#TODO 
'''
Za vse slike izračunaj modele in ostalo kar je potrebno 
'''

