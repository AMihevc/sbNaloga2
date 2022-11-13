from math import cos, pi, sin
import cv2
import glob
import numpy as np
import torch

#random things I need to test before implementing in solution

#naredi novo sliko z dodano resnico
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

#izriše podano sliko
def izrisi_sliko(slika):

    cv2.imshow('IZRIS SLIKE', slika)

    #kle morš pol prtisnt neko tipko da se zapre
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    return 0

#funkcija doda skatlo na sliko
#skatla mora biti oblike (x,y,w,h)
def dodaj_skatlo(slika, skatla):
    for (x,y,w,h) in skatla:
        cv2.rectangle(slika, (x, y),(x + w, y + h), (0, 0, 255), 5)
    return 0


#pravilna pot do slik in resnic 
pot = "Support Files/ear_data/test/"
absolutna_pot = "/home/anze/Documents/Faks/Slikovna biometrija/DrugaDomacaNaloga/sbNaloga2/Support Files/ear_data/test/"

poti_do_slik = glob.glob(pot + '*.png')

#preberi eno sliko

pot_do_slike = "Support Files/ear_data/test/0501.png" # mr.bean
slika = cv2.imread(pot_do_slike, 1)

#slika_rgb = cv2.cvtColor(slika,cv2.COLOR_BGR2RGB)

#mr bean z resnico 
slika_z_resnico = dodaj_resnico(pot_do_slike)

#izris slike 
izrisi_sliko(slika)

#izrisi_sliko(slika_z_resnico)

# yolov5 model
#model = torch.hub.load('Support Files', model='yolo5s', source='local')

model_yolo = torch.hub.load( 'yolov5', 'custom', path='Support Files/yolo5s.pt', source="local")

rezultat_yolo = model_yolo(slika)

rezultat_yolo.print()


#haar-cascade 

#sliko pretvori v gray scale 
slika_grey = cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY)

#naloži modele za levo in desno uho 

levo_uho_filter_VJ = cv2.CascadeClassifier('Support Files\haarcascade_mcs_leftear.xml')
desno_uho_filter_VJ = cv2.CascadeClassifier('Support Files\haarcascade_mcs_rightear.xml')

levo_uho_skatla = levo_uho_filter_VJ.detectMultiScale(slika_grey, scaleFactor = 1.2, minNeighbors = 5)
desno_uho_skatla = desno_uho_filter_VJ.detectMultiScale(slika_grey, scaleFactor = 1.2, minNeighbors = 5)


dodaj_skatlo(slika_grey,levo_uho_skatla)

izrisi_sliko(slika_grey)