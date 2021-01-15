import numpy
import cv2
import sys
import easygui

def run_euc(matrix_a, matrix_b):
    dist = numpy.sqrt(numpy.sum((numpy.transpose(matrix_a, (1, 2, 0)) - matrix_b)**2, axis=2))
    return 1.0 - dist/numpy.max(dist)

def OnLayerChange(layer):
    global hypercube, num_layers, current_layer
    current_layer = layer
    #cv2.imshow('Image', hypercube[layer])
    red_edge = cv2.getTrackbarPos('red_edge', 'Settings')
    if layer<=red_edge:
        h = numpy.full_like(hypercube[layer], int(135.0*(1-layer/red_edge)) )
        s = numpy.full_like(hypercube[layer], 255)
    else:
        h = numpy.full_like(hypercube[layer], 0 )
        s = numpy.full_like(hypercube[layer], 1)
    v = hypercube[layer]

    hsv = cv2.merge([h, s, v])
    hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('Image', hsv)
    draw_diagram()

def draw_diagram():
    global diag_data
    if type(diag_data) != type(None):
        col_width = min(1980//len(diag_data), 15)
        img = numpy.zeros((256, num_layers*col_width, 3), numpy.uint8)
        red_edge = cv2.getTrackbarPos('red_edge', 'Settings')
        for i in range(num_layers):
            h = diag_data[i]#int(self.hist[i])
            if i<=red_edge:
                cv2.rectangle(img, (i*col_width, 255), ((i+1)*col_width-2, 255-h), (int(135.0*(1 - i/red_edge) ), 255, 255), -1)
            else:
                cv2.rectangle(img, (i*col_width, 255), ((i+1)*col_width-2, 255-h), (0, 1, 255), -1)

        cv2.line(img, (current_layer*col_width, 10), (current_layer*col_width, 245), (128,1,255))
        cv2.line(img, ((current_layer+1)*col_width-1, 10), ((current_layer+1)*col_width-1, 245), (128,1,255))
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        cv2.imshow('hist', img)

def onmouse(event, x, y, flags, param):
    global hypercube, num_layers, current_layer, diag_data
    if event == cv2.EVENT_LBUTTONDOWN:
        #print(x, y)
        #col_width = 15
        diag_data = hypercube[:,y,x]
        draw_diagram()
    elif (event == cv2.EVENT_RBUTTONDOWN):
        cv2.imshow('Distances Map', run_euc(hypercube, hypercube[:, y, x]))

def OnRedEdgeChange(red_edge):
    layer = cv2.getTrackbarPos('layer', 'Settings')
    OnLayerChange(layer)

def create_new_pipeline():
    global hypercube, num_layers, current_layer
    fn=easygui.fileopenbox(msg='Открыть гиперкуб numpy', default='*.npy', filetypes=['*.npy',])
    if fn:
        hypercube = numpy.load(fn)
        num_layers = hypercube.shape[0]
        cv2.namedWindow( "Image" )
        cv2.namedWindow( "Settings" )
        cv2.setMouseCallback('Image', onmouse)
        cv2.createTrackbar('layer', 'Settings', 0, num_layers-1, OnLayerChange)
        cv2.createTrackbar('red_edge', 'Settings', num_layers//2, num_layers-1, OnRedEdgeChange)
        cv2.setTrackbarMin('red_edge', 'Settings',1)
        current_layer = 0
        OnLayerChange(0)

def save_diag_data():
    global diag_data
    fn = easygui.filesavebox(msg='Сохранить гистограмму точки в csv', default='*.csv', filetypes=['*.csv',])
    if fn:
        numpy.savetxt(fn, diag_data, 
            delimiter =", ",  
            fmt ='% s') 
    #print(fn)
    #for val in diag_data:
        #print(val)

def write_images():
    global hypercube
    fn = easygui.filesavebox(msg='Сохранить слои в png', default='img_.png', filetypes=['*.png',])
    if fn:
        # cv2.imwrite(fn,hypercube[0]) - эта хрень не понимает русских букв в пути
        num_layers = hypercube.shape[0]
        fn = fn.split('.')[0]
        for i in range(num_layers):
            retval, buf = cv2.imencode('.png', hypercube[i])
            buf.tofile(fn+str(i)+'.png')

def main():
    global diag_data
    create_new_pipeline()
    while True:
        ch = cv2.waitKey(5)
        if ch == 27:
            break
        elif ch == ord('o'):
            cv2.destroyAllWindows()
            create_new_pipeline()
        elif ch == ord('s'):
            save_diag_data()
        elif ch == ord('w'):
            write_images()
    cv2.destroyAllWindows()


diag_data = None
if __name__ == "__main__":
    main()

"""
import json
with open('tiscam.json', 'r') as myfile:
    data=myfile.read()

# parse file
obj = json.loads(data)
from pprint import pprint
pprint(obj)
print('Слоёв в кубе: ', len(obj['layers']))
print(obj['layers'][0]['name'])
"""