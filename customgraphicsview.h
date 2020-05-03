#ifndef CUSTOMGRAPHICSVIEW_H
#define CUSTOMGRAPHICSVIEW_H

#include <QGraphicsView>
#include <QWheelEvent>

class CustomGraphicsView : public QGraphicsView
{
    Q_OBJECT
public:
    CustomGraphicsView(QWidget *t_parent = nullptr);

public slots:
    //Resizes image to fit in view
    void fitToView();

private:
    //Modifies zoom by factor
    void zoom(const double factor);

protected:
    //Handles scrollwheel event
    void wheelEvent(QWheelEvent *event) override;
};

#endif // CUSTOMGRAPHICSVIEW_H
