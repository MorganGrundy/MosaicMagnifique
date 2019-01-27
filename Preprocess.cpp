#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <sys/ioctl.h>
#include <cstdio>
#include <unistd.h>
#include <boost/filesystem.hpp>
#include <math.h>

#include "shared.hpp"

using namespace cv;
using namespace std;
using namespace boost::filesystem;

//Reads all images in fn, resizes them, and writes into out path
//Provides progress bar, records time taken and outputs
//Complexity: O(Ni), where Ni = Number of images
void resize_images(vector<String> *fn, struct winsize w, String img_out_path)
{
  double t = getTickCount();

  //Read in images and preprocess them
  Mat temp_img;
  for (size_t i = 0; i < (*fn).size(); i++)
  {
      temp_img = imread((*fn)[i], IMREAD_COLOR);
      if (temp_img.empty())
      {
        cout << "Failed to load: " << (*fn)[i] << endl;
        return;
      }
      imageToSquare(temp_img);
      resizeImageExclusive(temp_img, temp_img, MAX_CELL_SIZE, MAX_CELL_SIZE);

      String file_out = img_out_path + (*fn)[i].substr((*fn)[i].find_last_of("/"), (*fn)[i].length());
      imwrite(file_out, temp_img);
      progressBar(i, (*fn).size() - 1, w.ws_col);
  }
  progressBarClean(w.ws_col);
  t = (getTickCount() - t) / getTickFrequency();
  cout << "Time passed in seconds for read: " << t << endl;
}

int main(int argc, char** argv)
{
  //Reads args
  if(argc < 3)
  {
      cout << argv[0] << " images_in_path images_out_path" << endl << endl;

      //Outputs the accepted image types
      cout << "Accepted image types:" << endl;
      ostringstream oss;
      copy(IMG_FORMATS.begin(), IMG_FORMATS.end()-1, ostream_iterator<String>(oss, ", "));
      oss << IMG_FORMATS.back();
      cout << oss.str() << endl << endl;

      //Outputs flags and descriptions
      cout << "Flags:" << endl;
      cout << "-cell_size x, -cs x: Uses the integer in the next argument (x) as the cell size in pixels" << endl;

      return -1;
  }
  if (argc > 3) //Reads flags
  {
    cout << "Argc: " << argc << endl;
    for (int i = 4; i < argc; ++i)
    {
      string flag = argv[i];
      if (i + 1 < argc) //Flags that require two arguments
      {
        string other = argv[i + 1];
        if (flag == "-cs" || flag == "-cell_size")
        {
          cout << "Cell size: " << stoi(other) << endl;
          CELL_SIZE = stoi(other);
          MAX_CELL_SIZE = CELL_SIZE * (MAX_ZOOM / MIN_ZOOM);
          i++;
        }
      }
    }
  }

  path img_in_path(argv[1]);
  path img_out_path(argv[2]);
  if (!exists(img_out_path) || !is_directory(img_out_path))
  {
    cout << argv[2] << " is not a directory" << endl;
    return false;
  }

  //Reads all image names from given directory
  vector<String> fn;
  if (!read_image_names(img_in_path, &fn))
    return 0;

  //Used to determine width of window for progress bar
  struct winsize w;
  ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);

  //Loads, resizes images, writes new images
  cout << "Processing " << fn.size() << " images at size " << MAX_CELL_SIZE << ":" << endl;
  resize_images(&fn, w, argv[2]);
  return 0;
}
