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

#funkcija doda skatlo na sliko
#skatla mora biti oblike (x1,y1,x2,y2)
def dodaj_skatlo(slika, skatla, barva):
    for (x1,y1,x2,y2) in skatla:
        cv2.rectangle(slika, (x1, y1),(x2, y2), barva, 5)
    return 0


def izracunaj_Haar(pot_do_slike, scaleFaktor, minSosedov):


    #odpri sliko v gray scale
    slika_grey =  cv2.imread(pot_do_slike, 0)

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


def yolo_model_skatle(pot_do_slike):
    model  = torch.hub.load( 'yolov5', 'custom', path='Support Files/yolo5s.pt', source="local")

    rezultat_yolo = model(pot_do_slike)
    skatle_yolo = []
    for skatla in rezultat_yolo.xyxy[0].tolist():
        x1, y1, x2, y2, sigurnost, _ = [float(i) for i in skatla]
        skatla_yolo = [round(x1), round(y1), round(x2), round(y2)]
        skatle_yolo.append(skatla_yolo)

    return skatle_yolo

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
#izrisi_sliko(slika)

#izrisi_sliko(slika_z_resnico)

# yolov5 model ----------------------------------------
#model = torch.hub.load('Support Files', model='yolo5s', source='local')


model_yolo = torch.hub.load( 'yolov5', 'custom', path='Support Files/yolo5s.pt', source="local")

rezultat_yolo = model_yolo(pot_do_slike)

rezultat_yolo.show()

#rezultat_yolo.print()

skatle_yolo = yolo_model_skatle(pot_do_slike)
print(skatle_yolo)
dodaj_skatlo(slika_z_resnico,skatle_yolo,(255,0,0))
#skatla od iou glej discord za format
#print(rezultat_yolo.xyxy[0].tolist())

#yolo lahko tut več 

#haar-cascade --------------------------------------------
'''
#sliko pretvori v gray scale 
slika_grey = cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY)

#naloži modele za levo in desno uho 

levo_uho_filter_VJ = cv2.CascadeClassifier('Support Files\haarcascade_mcs_leftear.xml')
desno_uho_filter_VJ = cv2.CascadeClassifier('Support Files\haarcascade_mcs_rightear.xml')

levo_uho_skatla = levo_uho_filter_VJ.detectMultiScale(slika_grey, scaleFactor = 1.2, minNeighbors = 5)
desno_uho_skatla = desno_uho_filter_VJ.detectMultiScale(slika_grey, scaleFactor = 1.1, minNeighbors = 5)

#print(levo_uho_skatla)
#print(desno_uho_skatla)

'''
#izračunam Haar skatle 

zdruzen_seznam = izracunaj_Haar(slika, 1.2, 5)
#print(zdruzen_seznam)
nove_oblike = pretvor_obliko(zdruzen_seznam)

#samo za izris
print(nove_oblike)
dodaj_skatlo(slika_z_resnico, nove_oblike, (0,0,255))
izrisi_sliko(slika_z_resnico)

#za računanje IOU 

resnica = dobi_resnico(pot_do_slike)

print(resnica)
#print(resnica)

iou = get_iou(resnica,nove_oblike[0])

print(iou)

#optimizacija pri 0.5 treshold za iou

