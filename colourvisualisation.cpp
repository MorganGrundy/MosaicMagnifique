#include "colourvisualisation.h"
#include "ui_colourvisualisation.h"

ColourVisualisation::ColourVisualisation(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::ColourVisualisation)
{
    ui->setupUi(this);
}

ColourVisualisation::~ColourVisualisation()
{
    delete ui;
}
