#include "bmpWrapper.h"
#include "bmpfile.h"
#include "stdlib.h"
#include "stdio.h"

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

uint8_t pixel_map(float val){
  return (uint8_t)( MIN(val * 255, 255));
}

int writeImage(char* fname, int width, int height, float* pixel_buff){
  bmpfile_t *bmp = bmp_create(width, height, 32);

  if (bmp == NULL){
    return BMP_CREATE_ERROR;
  }
  rgb_pixel_t pixel = {0, 0, 0, 0};
  
  for (int i = 0; i < width; i++){
    for (int j = 0; j < height; j++){
      int pix_ix = (i * height + j) * 3;
      
      pixel.red = pixel_map(pixel_buff[pix_ix]);
      pixel.blue = pixel_map(pixel_buff[pix_ix + 1]);
      pixel.green = pixel_map(pixel_buff[pix_ix + 2]);
      if (!bmp_set_pixel(bmp, i, j, pixel))
	return BMP_SET_FAILED;
    }
  }

  if (!bmp_save(bmp, fname)){
    return BMP_SAVE_ERROR;
  }
  bmp_destroy(bmp);
  return BMP_SUCCESS;
}

