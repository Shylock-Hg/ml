
def main():
    r = 0
    g = 0
    b = 0

    color24 = 0
    color16 = 0

    while True :
        print("Input 24 bits color value <0xFFFFFF> below:\r\n")
        color24 = int(input(),base=16)
        r = color24>>16
        g = (color24&0x00FF00)>>8
        b =0x0000FF&color24

        r >>= 3
        g >>= 2
        b >>= 3

        color16 = (r<<11)+(g<<5)+b

        print("16 bits color:{:X}".format(color16))


if '__main__' == __name__:
    main()
