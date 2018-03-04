#include <stdio.h>

int main(void){
	unsigned char r = 0;
	unsigned char g = 0;
	unsigned char b = 0;
	unsigned int color24 = 0;
	unsigned int color16 = 0;
	while(1){
		printf("Input 24 bits color value below:\r\n");
		scanf("%x",&color24);
		r = color24>>16;
		g = (color24&0x00FF00)>>8;
		b =(unsigned char)color24;

		r >>= 3;
		g >>= 2;
		b >>= 3;
	
		color16 = (r<<11)+(g<<5)+b;

		printf("16 bits color:%x\r\n",color16);

	}

}

