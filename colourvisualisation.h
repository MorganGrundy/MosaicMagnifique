#ifndef COLOURVISUALISATION_H
#define COLOURVISUALISATION_H

#include <QMainWindow>

namespace Ui {
class ColourVisualisation;
}

class ColourVisualisation : public QMainWindow
{
    Q_OBJECT

public:
    explicit ColourVisualisation(QWidget *parent = nullptr);
    ~ColourVisualisation();

private:
    Ui::ColourVisualisation *ui;
};

#endif // COLOURVISUALISATION_H
