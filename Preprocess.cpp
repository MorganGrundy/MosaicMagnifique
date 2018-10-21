#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <sys/ioctl.h>
#include <cstdio>
#include <unistd.h>
#include <boost/filesystem.hpp>

#include "shared.hpp"

using namespace cv;
using namespace std;
using namespace boost::filesystem;

void process_images(vector<String> *fn, struct winsize w, String img_out_path)
{
  double t = getTickCount();

  //Read in images and preprocess them
  Mat temp_img;
  int max_cell_size = CELL_SIZE * (MAX_ZOOM / 100.0);
  for (size_t i = 0; i < (*fn).size(); i++)
  {
      temp_img = imread((*fn)[i], IMREAD_COLOR);
      if (temp_img.empty())
      {
        cout << "Failed to load: " << (*fn)[i] << endl;
        return;
      }
      resizeImageExclusive(temp_img, temp_img, max_cell_size, max_cell_size);
      String file_out = img_out_path + (*fn)[i].substr((*fn)[i].find_last_of("/"), (*fn)[i].length());
      imwrite(file_out, temp_img);
      progressBar(i, (*fn).size() - 1, w.ws_col);
  }
  t = (getTickCount() - t) / getTickFrequency();
  cout << "\nTime passed in seconds for read: " << t << endl;
}

int main(int argc, char** argv)
{
    //Read args
    if(argc < 3)
    {
        cout << argv[0] << " images_in_path images_out_path" << endl;
        return -1;
    }
    path img_in_path(argv[1]);
    path img_out_path(argv[2]);
    if (!exists(img_out_path) || !is_directory(img_out_path))
      return false;

    vector<String> fn;
    if (!read_image_names(img_in_path, &fn))
      return 0;

    //Used to determine width of window for progress bar
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);

    cout << "Processing " << fn.size() << " images at size " << CELL_SIZE << ":" << endl;
    process_images(&fn, w, argv[2]);

    /*
    //Get list of photos in given folder
    vector<String> fn;
    String filepath(img_in_path);
    filepath += "*.jpg";
    glob(filepath, fn, false); //Populates fn with all the .jpg at filepath

    cout << "Processing " << fn.size() << " images at size " << CELL_SIZE << ":" << endl;

    process_images(fn, w);*/

    return 0;
}
