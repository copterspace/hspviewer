import numpy
import cv2
import sys
import easygui
import os

images_ext_list = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', 'RGB Image'] #Images list
compare_points = []

def run_euc(matrix_a, matrix_b):
    global diag_flags
    selections_list = [i for i in range(len(diag_flags)) if diag_flags[i]]
    dist = numpy.sqrt(numpy.sum((numpy.transpose(matrix_a, (1, 2, 0))[:, :, selections_list] - matrix_b[selections_list])**2, axis=2))
    temp_min = numpy.min(dist)
    return 1.0 - (dist - temp_min) / (numpy.max(dist) - temp_min)

def OnLayerChange(layer):
    global hypercube, num_layers, current_layer, source_frame, compare_points
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
    # draw compare points
    num_points = cv2.getTrackbarPos('num_points', 'Settings')
    while len(compare_points)>num_points:
        del compare_points[0]
    for i in range(len(compare_points)):
        x, y = compare_points[i]
        color_index = int(135.0*(i/num_points))
        cv2.line(hsv, (x-10,y), (x+10,y), (color_index,255,255),1)
        cv2.line(hsv, (x,y-10), (x,y+10), (color_index,255,255),1)
        cv2.putText(hsv, str(i+1), (x+2,y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (color_index,255,255), 1)
    hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    source_frame = hsv
    cv2.imshow('Image', hsv)
    draw_diagram()

def draw_diagram():
    global diag_flags, compare_points, hypercube, num_layers #diag_data, 
    #if type(diag_data) != type(None):
    num_points = len(compare_points)
    if num_points > 0:
        col_width = min(1980//num_layers, 15)
        img = numpy.zeros((256+12, num_layers*col_width, 3), numpy.uint8)
        red_edge = cv2.getTrackbarPos('red_edge', 'Settings')
        for pt in range(num_points):
            x, y = compare_points[pt]
            diag_data = hypercube[:, y, x]
            for i in range(num_layers):
                h = diag_data[i]#int(self.hist[i])
                if i <= red_edge:
                    cv2.rectangle(img, (i * col_width, 255), ((i + 1) * col_width - 2, 255 - h), (int(135.0*(1 - i/red_edge) ), 255, 255), -1)
                else:
                    cv2.rectangle(img, (i * col_width, 255), ((i + 1) * col_width - 2, 255 - h), (0, 1, 255), -1)
        # Draw layer graph
        for pt in range(num_points):
            x, y = compare_points[pt]
            diag_data = hypercube[:, y, x]
            color_index = int(135.0*(pt/num_points))
            for i in range(num_layers):
                if i > 0:
                    h = diag_data[i]
                    h0 = diag_data[i-1]
                    cv2.line(img, ((i-1)*col_width + col_width//2, 255 - h0), (i*col_width + col_width//2, 255-h), (color_index , 255, 255) ,2)
        # Draw layer checkbox
        for i in range(num_layers):
            if diag_flags[i]:
                cv2.line(img,(i * col_width+2, 262),((i + 1) * col_width - 2, 262),(80,255,255), 2)
                cv2.line(img,(i * col_width + col_width//2, 259),(i * col_width + col_width//2, 266),(80,255,255), 2)
            else:
                cv2.line(img,(i * col_width+2, 262),((i + 1) * col_width - 2, 262),(20,255,255), 2)
        cv2.line(img, (current_layer * col_width, 10), (current_layer*col_width, 245), (128, 1, 255))
        cv2.line(img, ((current_layer + 1) * col_width - 1, 10), ((current_layer + 1) * col_width - 1, 245), (128, 1, 255))
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        cv2.imshow('hist', img)
        cv2.setMouseCallback('hist', onmouse_diagram)

def onmouse(event, x, y, flags, param):
    global hypercube, num_layers, current_layer, diag_data, x0, y0, source_frame, paint_flag, compare_points
    
    if paint_flag:
        res_frame = source_frame.copy()
        cv2.rectangle(res_frame, (x0, y0), (x, y), (255, 255, 255), 1)
        cv2.imshow('Image', res_frame)        
        
    if (event == cv2.EVENT_LBUTTONDOWN):
        # diag_data = hypercube[:, y, x]
        compare_points.append( (x,y) )
        draw_diagram()
        OnLayerChange(current_layer)
    elif (event == cv2.EVENT_MBUTTONDOWN):
        cv2.imshow('Distances Map', run_euc(hypercube, hypercube[:, y, x]))
    elif (event == cv2.EVENT_RBUTTONDOWN):
        x0 = x
        y0 = y        
        paint_flag = True
    elif (event == cv2.EVENT_RBUTTONUP):
        paint_flag = False
        if not ((x == x0) or (y == y0)):
            cv2.imshow('Distances Map', run_euc(hypercube, numpy.mean(numpy.mean(hypercube[:, min(y0, y):max(y0, y), min(x0, x): max(x0, x)], axis=1), axis=1)))
        else:
            cv2.imshow('Distances Map', run_euc(hypercube, hypercube[:, y, x]))

def onmouse_diagram(event, x, y, flags, param):
    global num_layers, current_layer, diag_data, diag_flags, num_layers

    if (event == cv2.EVENT_LBUTTONUP):
        col_width = min(1980//num_layers, 15)
        num_col = x // col_width
        diag_flags[num_col] = not diag_flags[num_col] 
        draw_diagram()
    elif (event == cv2.EVENT_RBUTTONUP):
        for i in range(len(diag_flags)):
            diag_flags[i] = not diag_flags[i]
        draw_diagram()

def OnRedEdgeChange(red_edge):
    layer = cv2.getTrackbarPos('layer', 'Settings')
    OnLayerChange(layer)

def OnNumPointsChange(num_points):
    global current_layer, compare_points
    while num_points<len(compare_points):
        del compare_points[0]
    OnLayerChange(current_layer)

def create_new_pipeline():
    global hypercube, num_layers, current_layer, paint_flag, diag_flags
    paint_flag = False
    fn = easygui.fileopenbox(msg='Открыть гиперкуб numpy', filetypes=[['.npy', 'Numpy Hypercube'], ['.tiff', 'GeoTIFF'], images_ext_list], default='*.npy')
    if fn:
        _, ext_ = os.path.splitext(fn)        
        if (ext_ == '.npy'):
            hypercube = numpy.load(fn)
        elif (ext_ == '.tiff'):
            # https://kipcrossing.github.io/2021-01-04-geotiff-python-package/
            from geotiff import GeoTiff
            geoTiff = GeoTiff(fn)
            hypercube = geoTiff.read()[:].transpose((2, 0, 1))
        elif (ext_ in images_ext_list):
            #hypercube = cv2.imread(fn).transpose((2, 0, 1)) #EXIF fix might be here...            
            f = open(fn, "rb")
            img = cv2.imdecode(numpy.frombuffer(f.read(), dtype=numpy.uint8), cv2.IMREAD_COLOR)
            hypercube = img.transpose((2, 0, 1))
        num_layers = hypercube.shape[0]
        cv2.namedWindow( "Image" )
        cv2.namedWindow( "Settings", cv2.WINDOW_NORMAL )
        cv2.resizeWindow("Settings", 300, 100)
        cv2.setMouseCallback('Image', onmouse)
        cv2.createTrackbar('layer', 'Settings', 0, num_layers-1, OnLayerChange)
        cv2.createTrackbar('red_edge', 'Settings', num_layers//2, num_layers-1, OnRedEdgeChange)
        cv2.setTrackbarMin('red_edge', 'Settings', 1)
        cv2.createTrackbar('num_points', 'Settings', 1, 9, OnNumPointsChange)
        cv2.setTrackbarMin('num_points', 'Settings', 1)
        current_layer = 0
        diag_flags = [True for i in range(num_layers)]#numpy.ones((num_layers,), dtype=int)
        OnLayerChange(0)

def save_diag_data():
    global diag_data
    fn = easygui.filesavebox(msg='Сохранить гистограмму точки в csv', default='*.csv', filetypes=['*.csv',])
    if fn:
        numpy.savetxt(fn, diag_data,
            delimiter =", ",
            fmt ='% s')

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
