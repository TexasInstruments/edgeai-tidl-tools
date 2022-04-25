// Downloaded from https://repo.progsbase.com - Code Developed Using progsbase.

#include "../include/pbPlots.hpp"

using namespace std;

bool CropLineWithinBoundary(NumberReference *x1Ref, NumberReference *y1Ref, NumberReference *x2Ref, NumberReference *y2Ref, double xMin, double xMax, double yMin, double yMax){
  double x1, y1, x2, y2;
  bool success, p1In, p2In;
  double dx, dy, f1, f2, f3, f4, f;

  x1 = x1Ref->numberValue;
  y1 = y1Ref->numberValue;
  x2 = x2Ref->numberValue;
  y2 = y2Ref->numberValue;

  p1In = x1 >= xMin && x1 <= xMax && y1 >= yMin && y1 <= yMax;
  p2In = x2 >= xMin && x2 <= xMax && y2 >= yMin && y2 <= yMax;

  if(p1In && p2In){
    success = true;
  }else if( !p1In  && p2In){
    dx = x1 - x2;
    dy = y1 - y2;

    if(dx != 0.0){
      f1 = (xMin - x2)/dx;
      f2 = (xMax - x2)/dx;
    }else{
      f1 = 1.0;
      f2 = 1.0;
    }
    if(dy != 0.0){
      f3 = (yMin - y2)/dy;
      f4 = (yMax - y2)/dy;
    }else{
      f3 = 1.0;
      f4 = 1.0;
    }

    if(f1 < 0.0){
      f1 = 1.0;
    }
    if(f2 < 0.0){
      f2 = 1.0;
    }
    if(f3 < 0.0){
      f3 = 1.0;
    }
    if(f4 < 0.0){
      f4 = 1.0;
    }

    f = fmin(f1, fmin(f2, fmin(f3, f4)));

    x1 = x2 + f*dx;
    y1 = y2 + f*dy;

    success = true;
  }else if(p1In &&  !p2In ){
    dx = x2 - x1;
    dy = y2 - y1;

    if(dx != 0.0){
      f1 = (xMin - x1)/dx;
      f2 = (xMax - x1)/dx;
    }else{
      f1 = 1.0;
      f2 = 1.0;
    }
    if(dy != 0.0){
      f3 = (yMin - y1)/dy;
      f4 = (yMax - y1)/dy;
    }else{
      f3 = 1.0;
      f4 = 1.0;
    }

    if(f1 < 0.0){
      f1 = 1.0;
    }
    if(f2 < 0.0){
      f2 = 1.0;
    }
    if(f3 < 0.0){
      f3 = 1.0;
    }
    if(f4 < 0.0){
      f4 = 1.0;
    }

    f = fmin(f1, fmin(f2, fmin(f3, f4)));

    x2 = x1 + f*dx;
    y2 = y1 + f*dy;

    success = true;
  }else{
    success = false;
  }

  x1Ref->numberValue = x1;
  y1Ref->numberValue = y1;
  x2Ref->numberValue = x2;
  y2Ref->numberValue = y2;

  return success;
}
double IncrementFromCoordinates(double x1, double y1, double x2, double y2){
  return (x2 - x1)/(y2 - y1);
}
double InterceptFromCoordinates(double x1, double y1, double x2, double y2){
  double a, b;

  a = IncrementFromCoordinates(x1, y1, x2, y2);
  b = y1 - a*x1;

  return b;
}
vector<RGBA*> *Get8HighContrastColors(){
  vector<RGBA*> *colors;
  colors = new vector<RGBA*> (8.0);
  colors->at(0) = CreateRGBColor(3.0/256.0, 146.0/256.0, 206.0/256.0);
  colors->at(1) = CreateRGBColor(253.0/256.0, 83.0/256.0, 8.0/256.0);
  colors->at(2) = CreateRGBColor(102.0/256.0, 176.0/256.0, 50.0/256.0);
  colors->at(3) = CreateRGBColor(208.0/256.0, 234.0/256.0, 43.0/256.0);
  colors->at(4) = CreateRGBColor(167.0/256.0, 25.0/256.0, 75.0/256.0);
  colors->at(5) = CreateRGBColor(254.0/256.0, 254.0/256.0, 51.0/256.0);
  colors->at(6) = CreateRGBColor(134.0/256.0, 1.0/256.0, 175.0/256.0);
  colors->at(7) = CreateRGBColor(251.0/256.0, 153.0/256.0, 2.0/256.0);
  return colors;
}
void DrawFilledRectangleWithBorder(RGBABitmapImage *image, double x, double y, double w, double h, RGBA *borderColor, RGBA *fillColor){
  if(h > 0.0 && w > 0.0){
    DrawFilledRectangle(image, x, y, w, h, fillColor);
    DrawRectangle1px(image, x, y, w, h, borderColor);
  }
}
RGBABitmapImageReference *CreateRGBABitmapImageReference(){
  RGBABitmapImageReference *reference;

  reference = new RGBABitmapImageReference();
  reference->image = new RGBABitmapImage();
  reference->image->x = new vector<RGBABitmap*> (0.0);

  return reference;
}
bool RectanglesOverlap(Rectangle *r1, Rectangle *r2){
  bool overlap;

  overlap = false;

  overlap = overlap || (r2->x1 >= r1->x1 && r2->x1 <= r1->x2 && r2->y1 >= r1->y1 && r2->y1 <= r1->y2);
  overlap = overlap || (r2->x2 >= r1->x1 && r2->x2 <= r1->x2 && r2->y1 >= r1->y1 && r2->y1 <= r1->y2);
  overlap = overlap || (r2->x1 >= r1->x1 && r2->x1 <= r1->x2 && r2->y2 >= r1->y1 && r2->y2 <= r1->y2);
  overlap = overlap || (r2->x2 >= r1->x1 && r2->x2 <= r1->x2 && r2->y2 >= r1->y1 && r2->y2 <= r1->y2);

  return overlap;
}
Rectangle *CreateRectangle(double x1, double y1, double x2, double y2){
  Rectangle *r;
  r = new Rectangle();
  r->x1 = x1;
  r->y1 = y1;
  r->x2 = x2;
  r->y2 = y2;
  return r;
}
void CopyRectangleValues(Rectangle *rd, Rectangle *rs){
  rd->x1 = rs->x1;
  rd->y1 = rs->y1;
  rd->x2 = rs->x2;
  rd->y2 = rs->y2;
}
void DrawXLabelsForPriority(double p, double xMin, double oy, double xMax, double xPixelMin, double xPixelMax, NumberReference *nextRectangle, RGBA *gridLabelColor, RGBABitmapImage *canvas, vector<double> *xGridPositions, StringArrayReference *xLabels, NumberArrayReference *xLabelPriorities, vector<Rectangle*> *occupied, bool textOnBottom){
  bool overlap, currentOverlaps;
  double i, j, x, px, padding;
  vector<wchar_t> *text;
  Rectangle *r;

  r = new Rectangle();
  padding = 10.0;

  overlap = false;
  for(i = 0.0; i < xLabels->stringArray->size(); i = i + 1.0){
    if(xLabelPriorities->numberArray->at(i) == p){

      x = xGridPositions->at(i);
      px = MapXCoordinate(x, xMin, xMax, xPixelMin, xPixelMax);
      text = xLabels->stringArray->at(i)->string;

      r->x1 = floor(px - GetTextWidth(text)/2.0);
      if(textOnBottom){
        r->y1 = floor(oy + 5.0);
      }else{
        r->y1 = floor(oy - 20.0);
      }
      r->x2 = r->x1 + GetTextWidth(text);
      r->y2 = r->y1 + GetTextHeight(text);

      /* Add padding */
      r->x1 = r->x1 - padding;
      r->y1 = r->y1 - padding;
      r->x2 = r->x2 + padding;
      r->y2 = r->y2 + padding;

      currentOverlaps = false;

      for(j = 0.0; j < nextRectangle->numberValue; j = j + 1.0){
        currentOverlaps = currentOverlaps || RectanglesOverlap(r, occupied->at(j));
      }

      if( !currentOverlaps  && p == 1.0){
        DrawText(canvas, r->x1 + padding, r->y1 + padding, text, gridLabelColor);

        CopyRectangleValues(occupied->at(nextRectangle->numberValue), r);
        nextRectangle->numberValue = nextRectangle->numberValue + 1.0;
      }

      overlap = overlap || currentOverlaps;
    }
  }
  if( !overlap  && p != 1.0){
    for(i = 0.0; i < xGridPositions->size(); i = i + 1.0){
      x = xGridPositions->at(i);
      px = MapXCoordinate(x, xMin, xMax, xPixelMin, xPixelMax);

      if(xLabelPriorities->numberArray->at(i) == p){
        text = xLabels->stringArray->at(i)->string;

        r->x1 = floor(px - GetTextWidth(text)/2.0);
        if(textOnBottom){
          r->y1 = floor(oy + 5.0);
        }else{
          r->y1 = floor(oy - 20.0);
        }
        r->x2 = r->x1 + GetTextWidth(text);
        r->y2 = r->y1 + GetTextHeight(text);

        DrawText(canvas, r->x1, r->y1, text, gridLabelColor);

        CopyRectangleValues(occupied->at(nextRectangle->numberValue), r);
        nextRectangle->numberValue = nextRectangle->numberValue + 1.0;
      }
    }
  }
}
void DrawYLabelsForPriority(double p, double yMin, double ox, double yMax, double yPixelMin, double yPixelMax, NumberReference *nextRectangle, RGBA *gridLabelColor, RGBABitmapImage *canvas, vector<double> *yGridPositions, StringArrayReference *yLabels, NumberArrayReference *yLabelPriorities, vector<Rectangle*> *occupied, bool textOnLeft){
  bool overlap, currentOverlaps;
  double i, j, y, py, padding;
  vector<wchar_t> *text;
  Rectangle *r;

  r = new Rectangle();
  padding = 10.0;

  overlap = false;
  for(i = 0.0; i < yLabels->stringArray->size(); i = i + 1.0){
    if(yLabelPriorities->numberArray->at(i) == p){

      y = yGridPositions->at(i);
      py = MapYCoordinate(y, yMin, yMax, yPixelMin, yPixelMax);
      text = yLabels->stringArray->at(i)->string;

      if(textOnLeft){
        r->x1 = floor(ox - GetTextWidth(text) - 10.0);
      }else{
        r->x1 = floor(ox + 10.0);
      }
      r->y1 = floor(py - 6.0);
      r->x2 = r->x1 + GetTextWidth(text);
      r->y2 = r->y1 + GetTextHeight(text);

      /* Add padding */
      r->x1 = r->x1 - padding;
      r->y1 = r->y1 - padding;
      r->x2 = r->x2 + padding;
      r->y2 = r->y2 + padding;

      currentOverlaps = false;

      for(j = 0.0; j < nextRectangle->numberValue; j = j + 1.0){
        currentOverlaps = currentOverlaps || RectanglesOverlap(r, occupied->at(j));
      }

      /* Draw labels with priority 1 if they do not overlap anything else. */
      if( !currentOverlaps  && p == 1.0){
        DrawText(canvas, r->x1 + padding, r->y1 + padding, text, gridLabelColor);

        CopyRectangleValues(occupied->at(nextRectangle->numberValue), r);
        nextRectangle->numberValue = nextRectangle->numberValue + 1.0;
      }

      overlap = overlap || currentOverlaps;
    }
  }
  if( !overlap  && p != 1.0){
    for(i = 0.0; i < yGridPositions->size(); i = i + 1.0){
      y = yGridPositions->at(i);
      py = MapYCoordinate(y, yMin, yMax, yPixelMin, yPixelMax);

      if(yLabelPriorities->numberArray->at(i) == p){
        text = yLabels->stringArray->at(i)->string;

        if(textOnLeft){
          r->x1 = floor(ox - GetTextWidth(text) - 10.0);
        }else{
          r->x1 = floor(ox + 10.0);
        }
        r->y1 = floor(py - 6.0);
        r->x2 = r->x1 + GetTextWidth(text);
        r->y2 = r->y1 + GetTextHeight(text);

        DrawText(canvas, r->x1, r->y1, text, gridLabelColor);

        CopyRectangleValues(occupied->at(nextRectangle->numberValue), r);
        nextRectangle->numberValue = nextRectangle->numberValue + 1.0;
      }
    }
  }
}
vector<double> *ComputeGridLinePositions(double cMin, double cMax, StringArrayReference *labels, NumberArrayReference *priorities){
  vector<double> *positions;
  double cLength, p, pMin, pMax, pInterval, pNum, i, num, rem, priority, mode;

  cLength = cMax - cMin;

  p = floor(log10(cLength));
  pInterval = pow(10.0, p);
  /* gives 10-1 lines for 100-10 diff */
  pMin = ceil(cMin/pInterval)*pInterval;
  pMax = floor(cMax/pInterval)*pInterval;
  pNum = Round((pMax - pMin)/pInterval + 1.0);

  mode = 1.0;

  if(pNum <= 3.0){
    p = floor(log10(cLength) - 1.0);
    /* gives 100-10 lines for 100-10 diff */
    pInterval = pow(10.0, p);
    pMin = ceil(cMin/pInterval)*pInterval;
    pMax = floor(cMax/pInterval)*pInterval;
    pNum = Round((pMax - pMin)/pInterval + 1.0);

    mode = 4.0;
  }else if(pNum <= 6.0){
    p = floor(log10(cLength));
    pInterval = pow(10.0, p)/4.0;
    /* gives 40-5 lines for 100-10 diff */
    pMin = ceil(cMin/pInterval)*pInterval;
    pMax = floor(cMax/pInterval)*pInterval;
    pNum = Round((pMax - pMin)/pInterval + 1.0);

    mode = 3.0;
  }else if(pNum <= 10.0){
    p = floor(log10(cLength));
    pInterval = pow(10.0, p)/2.0;
    /* gives 20-3 lines for 100-10 diff */
    pMin = ceil(cMin/pInterval)*pInterval;
    pMax = floor(cMax/pInterval)*pInterval;
    pNum = Round((pMax - pMin)/pInterval + 1.0);

    mode = 2.0;
  }

  positions = new vector<double> (pNum);
  labels->stringArray = new vector<StringReference*> (pNum);
  priorities->numberArray = new vector<double> (pNum);

  for(i = 0.0; i < pNum; i = i + 1.0){
    num = pMin + pInterval*i;
    positions->at(i) = num;

    /* Always print priority 1 labels. Only draw priority 2 if they can all be drawn. Then, only draw priority 3 if they can all be drawn. */
    priority = 1.0;

    /* Prioritize x.25, x.5 and x.75 lower. */
    if(mode == 2.0 || mode == 3.0){
      rem = fmod(abs(round(num/pow(10.0, p - 2.0))), 100.0);

      priority = 1.0;
      if(rem == 50.0){
        priority = 2.0;
      }else if(rem == 25.0 || rem == 75.0){
        priority = 3.0;
      }
    }

    /* Prioritize x.1-x.4 and x.6-x.9 lower */
    if(mode == 4.0){
      rem = fmod(abs(Round(num/pow(10.0, p))), 10.0);

      priority = 1.0;
      if(rem == 1.0 || rem == 2.0 || rem == 3.0 || rem == 4.0 || rem == 6.0 || rem == 7.0 || rem == 8.0 || rem == 9.0){
        priority = 2.0;
      }
    }

    /* 0 has lowest priority. */
    if(EpsilonCompare(num, 0.0, pow(10.0, p - 5.0))){
      priority = 3.0;
    }

    priorities->numberArray->at(i) = priority;

    /* The label itself. */
    labels->stringArray->at(i) = new StringReference();
    if(p < 0.0){
      if(mode == 2.0 || mode == 3.0){
        num = RoundToDigits(num,  -(p - 1.0));
      }else{
        num = RoundToDigits(num,  -p);
      }
    }
    labels->stringArray->at(i)->string = CreateStringDecimalFromNumber(num);
  }

  return positions;
}
double MapYCoordinate(double y, double yMin, double yMax, double yPixelMin, double yPixelMax){
  double yLength, yPixelLength;

  yLength = yMax - yMin;
  yPixelLength = yPixelMax - yPixelMin;

  y = y - yMin;
  y = y*yPixelLength/yLength;
  y = yPixelLength - y;
  y = y + yPixelMin;
  return y;
}
double MapXCoordinate(double x, double xMin, double xMax, double xPixelMin, double xPixelMax){
  double xLength, xPixelLength;

  xLength = xMax - xMin;
  xPixelLength = xPixelMax - xPixelMin;

  x = x - xMin;
  x = x*xPixelLength/xLength;
  x = x + xPixelMin;
  return x;
}
double MapXCoordinateAutoSettings(double x, RGBABitmapImage *image, vector<double> *xs){
  return MapXCoordinate(x, GetMinimum(xs), GetMaximum(xs), GetDefaultPaddingPercentage()*ImageWidth(image), (1.0 - GetDefaultPaddingPercentage())*ImageWidth(image));
}
double MapYCoordinateAutoSettings(double y, RGBABitmapImage *image, vector<double> *ys){
  return MapYCoordinate(y, GetMinimum(ys), GetMaximum(ys), GetDefaultPaddingPercentage()*ImageHeight(image), (1.0 - GetDefaultPaddingPercentage())*ImageHeight(image));
}
double MapXCoordinateBasedOnSettings(double x, ScatterPlotSettings *settings){
  double xMin, xMax, xPadding, xPixelMin, xPixelMax;
  Rectangle *boundaries;

  boundaries = new Rectangle();
  ComputeBoundariesBasedOnSettings(settings, boundaries);
  xMin = boundaries->x1;
  xMax = boundaries->x2;

  if(settings->autoPadding){
    xPadding = floor(GetDefaultPaddingPercentage()*settings->width);
  }else{
    xPadding = settings->xPadding;
  }

  xPixelMin = xPadding;
  xPixelMax = settings->width - xPadding;

  return MapXCoordinate(x, xMin, xMax, xPixelMin, xPixelMax);
}
double MapYCoordinateBasedOnSettings(double y, ScatterPlotSettings *settings){
  double yMin, yMax, yPadding, yPixelMin, yPixelMax;
  Rectangle *boundaries;

  boundaries = new Rectangle();
  ComputeBoundariesBasedOnSettings(settings, boundaries);
  yMin = boundaries->y1;
  yMax = boundaries->y2;

  if(settings->autoPadding){
    yPadding = floor(GetDefaultPaddingPercentage()*settings->height);
  }else{
    yPadding = settings->yPadding;
  }

  yPixelMin = yPadding;
  yPixelMax = settings->height - yPadding;

  return MapYCoordinate(y, yMin, yMax, yPixelMin, yPixelMax);
}
double GetDefaultPaddingPercentage(){
  return 0.10;
}
void DrawText(RGBABitmapImage *canvas, double x, double y, vector<wchar_t> *text, RGBA *color){
  double i, charWidth, spacing;

  charWidth = 8.0;
  spacing = 2.0;

  for(i = 0.0; i < text->size(); i = i + 1.0){
    DrawAsciiCharacter(canvas, x + i*(charWidth + spacing), y, text->at(i), color);
  }
}
void DrawTextUpwards(RGBABitmapImage *canvas, double x, double y, vector<wchar_t> *text, RGBA *color){
  RGBABitmapImage *buffer, *rotated;

  buffer = CreateImage(GetTextWidth(text), GetTextHeight(text), GetTransparent());
  DrawText(buffer, 0.0, 0.0, text, color);
  rotated = RotateAntiClockwise90Degrees(buffer);
  DrawImageOnImage(canvas, rotated, x, y);
  DeleteImage(buffer);
  DeleteImage(rotated);
}
ScatterPlotSettings *GetDefaultScatterPlotSettings(){
  ScatterPlotSettings *settings;

  settings = new ScatterPlotSettings();

  settings->autoBoundaries = true;
  settings->xMax = 0.0;
  settings->xMin = 0.0;
  settings->yMax = 0.0;
  settings->yMin = 0.0;
  settings->autoPadding = true;
  settings->xPadding = 0.0;
  settings->yPadding = 0.0;
  settings->title = toVector(L"");
  settings->xLabel = toVector(L"");
  settings->yLabel = toVector(L"");
  settings->scatterPlotSeries = new vector<ScatterPlotSeries*> (0.0);
  settings->showGrid = true;
  settings->gridColor = GetGray(0.1);
  settings->xAxisAuto = true;
  settings->xAxisTop = false;
  settings->xAxisBottom = false;
  settings->yAxisAuto = true;
  settings->yAxisLeft = false;
  settings->yAxisRight = false;

  return settings;
}
ScatterPlotSeries *GetDefaultScatterPlotSeriesSettings(){
  ScatterPlotSeries *series;

  series = new ScatterPlotSeries();

  series->linearInterpolation = true;
  series->pointType = toVector(L"pixels");
  series->lineType = toVector(L"solid");
  series->lineThickness = 1.0;
  series->xs = new vector<double> (0.0);
  series->ys = new vector<double> (0.0);
  series->color = GetBlack();

  return series;
}
bool DrawScatterPlot(RGBABitmapImageReference *canvasReference, double width, double height, vector<double> *xs, vector<double> *ys, StringReference *errorMessage){
  ScatterPlotSettings *settings;
  bool success;

  settings = GetDefaultScatterPlotSettings();

  settings->width = width;
  settings->height = height;
  settings->scatterPlotSeries = new vector<ScatterPlotSeries*> (1.0);
  settings->scatterPlotSeries->at(0) = GetDefaultScatterPlotSeriesSettings();
  delete settings->scatterPlotSeries->at(0)->xs;
  settings->scatterPlotSeries->at(0)->xs = xs;
  delete settings->scatterPlotSeries->at(0)->ys;
  settings->scatterPlotSeries->at(0)->ys = ys;

  success = DrawScatterPlotFromSettings(canvasReference, settings, errorMessage);

  return success;
}
bool DrawScatterPlotFromSettings(RGBABitmapImageReference *canvasReference, ScatterPlotSettings *settings, StringReference *errorMessage){
  double xMin, xMax, yMin, yMax, xLength, yLength, i, x, y, xPrev, yPrev, px, py, pxPrev, pyPrev, originX, originY, p, l, plot;
  Rectangle *boundaries;
  double xPadding, yPadding, originXPixels, originYPixels;
  double xPixelMin, yPixelMin, xPixelMax, yPixelMax, xLengthPixels, yLengthPixels, axisLabelPadding;
  NumberReference *nextRectangle, *x1Ref, *y1Ref, *x2Ref, *y2Ref, *patternOffset;
  bool prevSet, success;
  RGBA *gridLabelColor;
  RGBABitmapImage *canvas;
  vector<double> *xs, *ys;
  bool linearInterpolation;
  ScatterPlotSeries *sp;
  vector<double> *xGridPositions, *yGridPositions;
  StringArrayReference *xLabels, *yLabels;
  NumberArrayReference *xLabelPriorities, *yLabelPriorities;
  vector<Rectangle*> *occupied;
  vector<bool> *linePattern;
  bool originXInside, originYInside, textOnLeft, textOnBottom;
  double originTextX, originTextY, originTextXPixels, originTextYPixels, side;

  canvas = CreateImage(settings->width, settings->height, GetWhite());
  patternOffset = CreateNumberReference(0.0);

  success = ScatterPlotFromSettingsValid(settings, errorMessage);

  if(success){

    boundaries = new Rectangle();
    ComputeBoundariesBasedOnSettings(settings, boundaries);
    xMin = boundaries->x1;
    yMin = boundaries->y1;
    xMax = boundaries->x2;
    yMax = boundaries->y2;

    /* If zero, set to defaults. */
    if(xMin - xMax == 0.0){
      xMin = 0.0;
      xMax = 10.0;
    }

    if(yMin - yMax == 0.0){
      yMin = 0.0;
      yMax = 10.0;
    }

    xLength = xMax - xMin;
    yLength = yMax - yMin;

    if(settings->autoPadding){
      xPadding = floor(GetDefaultPaddingPercentage()*settings->width);
      yPadding = floor(GetDefaultPaddingPercentage()*settings->height);
    }else{
      xPadding = settings->xPadding;
      yPadding = settings->yPadding;
    }

    /* Draw title */
    DrawText(canvas, floor(settings->width/2.0 - GetTextWidth(settings->title)/2.0), floor(yPadding/3.0), settings->title, GetBlack());

    /* Draw grid */
    xPixelMin = xPadding;
    yPixelMin = yPadding;
    xPixelMax = settings->width - xPadding;
    yPixelMax = settings->height - yPadding;
    xLengthPixels = xPixelMax - xPixelMin;
    yLengthPixels = yPixelMax - yPixelMin;
    DrawRectangle1px(canvas, xPixelMin, yPixelMin, xLengthPixels, yLengthPixels, settings->gridColor);

    gridLabelColor = GetGray(0.5);

    xLabels = new StringArrayReference();
    xLabelPriorities = new NumberArrayReference();
    yLabels = new StringArrayReference();
    yLabelPriorities = new NumberArrayReference();
    xGridPositions = ComputeGridLinePositions(xMin, xMax, xLabels, xLabelPriorities);
    yGridPositions = ComputeGridLinePositions(yMin, yMax, yLabels, yLabelPriorities);

    if(settings->showGrid){
      /* X-grid */
      for(i = 0.0; i < xGridPositions->size(); i = i + 1.0){
        x = xGridPositions->at(i);
        px = MapXCoordinate(x, xMin, xMax, xPixelMin, xPixelMax);
        DrawLine1px(canvas, px, yPixelMin, px, yPixelMax, settings->gridColor);
      }

      /* Y-grid */
      for(i = 0.0; i < yGridPositions->size(); i = i + 1.0){
        y = yGridPositions->at(i);
        py = MapYCoordinate(y, yMin, yMax, yPixelMin, yPixelMax);
        DrawLine1px(canvas, xPixelMin, py, xPixelMax, py, settings->gridColor);
      }
    }

    /* Compute origin information. */
    originYInside = yMin < 0.0 && yMax > 0.0;
    originY = 0.0;
    if(settings->xAxisAuto){
      if(originYInside){
        originY = 0.0;
      }else{
        originY = yMin;
      }
    }else{
if(settings->xAxisTop){
        originY = yMax;
      }
      if(settings->xAxisBottom){
        originY = yMin;
      }
    }
    originYPixels = MapYCoordinate(originY, yMin, yMax, yPixelMin, yPixelMax);

    originXInside = xMin < 0.0 && xMax > 0.0;
    originX = 0.0;
    if(settings->yAxisAuto){
      if(originXInside){
        originX = 0.0;
      }else{
        originX = xMin;
      }
    }else{
if(settings->yAxisLeft){
        originX = xMin;
      }
      if(settings->yAxisRight){
        originX = xMax;
      }
    }
    originXPixels = MapXCoordinate(originX, xMin, xMax, xPixelMin, xPixelMax);

    if(originYInside){
      originTextY = 0.0;
    }else{
      originTextY = yMin + yLength/2.0;
    }
    originTextYPixels = MapYCoordinate(originTextY, yMin, yMax, yPixelMin, yPixelMax);

    if(originXInside){
      originTextX = 0.0;
    }else{
      originTextX = xMin + xLength/2.0;
    }
    originTextXPixels = MapXCoordinate(originTextX, xMin, xMax, xPixelMin, xPixelMax);

    /* Labels */
    occupied = new vector<Rectangle*> (xLabels->stringArray->size() + yLabels->stringArray->size());
    for(i = 0.0; i < occupied->size(); i = i + 1.0){
      occupied->at(i) = CreateRectangle(0.0, 0.0, 0.0, 0.0);
    }
    nextRectangle = CreateNumberReference(0.0);

    /* x labels */
    for(i = 1.0; i <= 5.0; i = i + 1.0){
      textOnBottom = true;
      if( !settings->xAxisAuto  && settings->xAxisTop){
        textOnBottom = false;
      }
      DrawXLabelsForPriority(i, xMin, originYPixels, xMax, xPixelMin, xPixelMax, nextRectangle, gridLabelColor, canvas, xGridPositions, xLabels, xLabelPriorities, occupied, textOnBottom);
    }

    /* y labels */
    for(i = 1.0; i <= 5.0; i = i + 1.0){
      textOnLeft = true;
      if( !settings->yAxisAuto  && settings->yAxisRight){
        textOnLeft = false;
      }
      DrawYLabelsForPriority(i, yMin, originXPixels, yMax, yPixelMin, yPixelMax, nextRectangle, gridLabelColor, canvas, yGridPositions, yLabels, yLabelPriorities, occupied, textOnLeft);
    }

    /* Draw origin line axis titles. */
    axisLabelPadding = 20.0;

    /* x origin line */
    if(originYInside){
      DrawLine1px(canvas, Round(xPixelMin), Round(originYPixels), Round(xPixelMax), Round(originYPixels), GetBlack());
    }

    /* y origin line */
    if(originXInside){
      DrawLine1px(canvas, Round(originXPixels), Round(yPixelMin), Round(originXPixels), Round(yPixelMax), GetBlack());
    }

    /* Draw origin axis titles. */
    DrawTextUpwards(canvas, 10.0, floor(originTextYPixels - GetTextWidth(settings->yLabel)/2.0), settings->yLabel, GetBlack());
    DrawText(canvas, floor(originTextXPixels - GetTextWidth(settings->xLabel)/2.0), yPixelMax + axisLabelPadding, settings->xLabel, GetBlack());

    /* X-grid-markers */
    for(i = 0.0; i < xGridPositions->size(); i = i + 1.0){
      x = xGridPositions->at(i);
      px = MapXCoordinate(x, xMin, xMax, xPixelMin, xPixelMax);
      p = xLabelPriorities->numberArray->at(i);
      l = 1.0;
      if(p == 1.0){
        l = 8.0;
      }else if(p == 2.0){
        l = 3.0;
      }
      side =  -1.0;
      if( !settings->xAxisAuto  && settings->xAxisTop){
        side = 1.0;
      }
      DrawLine1px(canvas, px, originYPixels, px, originYPixels + side*l, GetBlack());
    }

    /* Y-grid-markers */
    for(i = 0.0; i < yGridPositions->size(); i = i + 1.0){
      y = yGridPositions->at(i);
      py = MapYCoordinate(y, yMin, yMax, yPixelMin, yPixelMax);
      p = yLabelPriorities->numberArray->at(i);
      l = 1.0;
      if(p == 1.0){
        l = 8.0;
      }else if(p == 2.0){
        l = 3.0;
      }
      side = 1.0;
      if( !settings->yAxisAuto  && settings->yAxisRight){
        side =  -1.0;
      }
      DrawLine1px(canvas, originXPixels, py, originXPixels + side*l, py, GetBlack());
    }

    /* Draw points */
    for(plot = 0.0; plot < settings->scatterPlotSeries->size(); plot = plot + 1.0){
      sp = settings->scatterPlotSeries->at(plot);

      xs = sp->xs;
      ys = sp->ys;
      linearInterpolation = sp->linearInterpolation;

      x1Ref = new NumberReference();
      y1Ref = new NumberReference();
      x2Ref = new NumberReference();
      y2Ref = new NumberReference();
      if(linearInterpolation){
        prevSet = false;
        xPrev = 0.0;
        yPrev = 0.0;
        for(i = 0.0; i < xs->size(); i = i + 1.0){
          x = xs->at(i);
          y = ys->at(i);

          if(prevSet){
            x1Ref->numberValue = xPrev;
            y1Ref->numberValue = yPrev;
            x2Ref->numberValue = x;
            y2Ref->numberValue = y;

            success = CropLineWithinBoundary(x1Ref, y1Ref, x2Ref, y2Ref, xMin, xMax, yMin, yMax);

            if(success){
              pxPrev = floor(MapXCoordinate(x1Ref->numberValue, xMin, xMax, xPixelMin, xPixelMax));
              pyPrev = floor(MapYCoordinate(y1Ref->numberValue, yMin, yMax, yPixelMin, yPixelMax));
              px = floor(MapXCoordinate(x2Ref->numberValue, xMin, xMax, xPixelMin, xPixelMax));
              py = floor(MapYCoordinate(y2Ref->numberValue, yMin, yMax, yPixelMin, yPixelMax));

              if(aStringsEqual(sp->lineType, toVector(L"solid")) && sp->lineThickness == 1.0){
                DrawLine1px(canvas, pxPrev, pyPrev, px, py, sp->color);
              }else if(aStringsEqual(sp->lineType, toVector(L"solid"))){
                DrawLine(canvas, pxPrev, pyPrev, px, py, sp->lineThickness, sp->color);
              }else if(aStringsEqual(sp->lineType, toVector(L"dashed"))){
                linePattern = GetLinePattern1();
                DrawLineBresenhamsAlgorithmThickPatterned(canvas, pxPrev, pyPrev, px, py, sp->lineThickness, linePattern, patternOffset, sp->color);
              }else if(aStringsEqual(sp->lineType, toVector(L"dotted"))){
                linePattern = GetLinePattern2();
                DrawLineBresenhamsAlgorithmThickPatterned(canvas, pxPrev, pyPrev, px, py, sp->lineThickness, linePattern, patternOffset, sp->color);
              }else if(aStringsEqual(sp->lineType, toVector(L"dotdash"))){
                linePattern = GetLinePattern3();
                DrawLineBresenhamsAlgorithmThickPatterned(canvas, pxPrev, pyPrev, px, py, sp->lineThickness, linePattern, patternOffset, sp->color);
              }else if(aStringsEqual(sp->lineType, toVector(L"longdash"))){
                linePattern = GetLinePattern4();
                DrawLineBresenhamsAlgorithmThickPatterned(canvas, pxPrev, pyPrev, px, py, sp->lineThickness, linePattern, patternOffset, sp->color);
              }else if(aStringsEqual(sp->lineType, toVector(L"twodash"))){
                linePattern = GetLinePattern5();
                DrawLineBresenhamsAlgorithmThickPatterned(canvas, pxPrev, pyPrev, px, py, sp->lineThickness, linePattern, patternOffset, sp->color);
              }
            }
          }

          prevSet = true;
          xPrev = x;
          yPrev = y;
        }
      }else{
        for(i = 0.0; i < xs->size(); i = i + 1.0){
          x = xs->at(i);
          y = ys->at(i);

          if(x > xMin && x < xMax && y > yMin && y < yMax){

            x = floor(MapXCoordinate(x, xMin, xMax, xPixelMin, xPixelMax));
            y = floor(MapYCoordinate(y, yMin, yMax, yPixelMin, yPixelMax));

            if(aStringsEqual(sp->pointType, toVector(L"crosses"))){
              DrawPixel(canvas, x, y, sp->color);
              DrawPixel(canvas, x + 1.0, y, sp->color);
              DrawPixel(canvas, x + 2.0, y, sp->color);
              DrawPixel(canvas, x - 1.0, y, sp->color);
              DrawPixel(canvas, x - 2.0, y, sp->color);
              DrawPixel(canvas, x, y + 1.0, sp->color);
              DrawPixel(canvas, x, y + 2.0, sp->color);
              DrawPixel(canvas, x, y - 1.0, sp->color);
              DrawPixel(canvas, x, y - 2.0, sp->color);
            }else if(aStringsEqual(sp->pointType, toVector(L"circles"))){
              DrawCircle(canvas, x, y, 3.0, sp->color);
            }else if(aStringsEqual(sp->pointType, toVector(L"dots"))){
              DrawFilledCircle(canvas, x, y, 3.0, sp->color);
            }else if(aStringsEqual(sp->pointType, toVector(L"triangles"))){
              DrawTriangle(canvas, x, y, 3.0, sp->color);
            }else if(aStringsEqual(sp->pointType, toVector(L"filled triangles"))){
              DrawFilledTriangle(canvas, x, y, 3.0, sp->color);
            }else if(aStringsEqual(sp->pointType, toVector(L"pixels"))){
              DrawPixel(canvas, x, y, sp->color);
            }
          }
        }
      }
    }

    canvasReference->image = canvas;
  }

  return success;
}
void ComputeBoundariesBasedOnSettings(ScatterPlotSettings *settings, Rectangle *boundaries){
  ScatterPlotSeries *sp;
  double plot, xMin, xMax, yMin, yMax;

  if(settings->scatterPlotSeries->size() >= 1.0){
    xMin = GetMinimum(settings->scatterPlotSeries->at(0)->xs);
    xMax = GetMaximum(settings->scatterPlotSeries->at(0)->xs);
    yMin = GetMinimum(settings->scatterPlotSeries->at(0)->ys);
    yMax = GetMaximum(settings->scatterPlotSeries->at(0)->ys);
  }else{
    xMin =  -10.0;
    xMax = 10.0;
    yMin =  -10.0;
    yMax = 10.0;
  }

  if( !settings->autoBoundaries ){
    xMin = settings->xMin;
    xMax = settings->xMax;
    yMin = settings->yMin;
    yMax = settings->yMax;
  }else{
    for(plot = 1.0; plot < settings->scatterPlotSeries->size(); plot = plot + 1.0){
      sp = settings->scatterPlotSeries->at(plot);

      xMin = fmin(xMin, GetMinimum(sp->xs));
      xMax = fmax(xMax, GetMaximum(sp->xs));
      yMin = fmin(yMin, GetMinimum(sp->ys));
      yMax = fmax(yMax, GetMaximum(sp->ys));
    }
  }

  boundaries->x1 = xMin;
  boundaries->y1 = yMin;
  boundaries->x2 = xMax;
  boundaries->y2 = yMax;
}
bool ScatterPlotFromSettingsValid(ScatterPlotSettings *settings, StringReference *errorMessage){
  bool success, found;
  ScatterPlotSeries *series;
  double i;

  success = true;

  /* Check axis placement. */
  if( !settings->xAxisAuto ){
    if(settings->xAxisTop && settings->xAxisBottom){
      success = false;
      errorMessage->string = toVector(L"x-axis not automatic and configured to be both on top and on bottom.");
    }
    if( !settings->xAxisTop  &&  !settings->xAxisBottom ){
      success = false;
      errorMessage->string = toVector(L"x-axis not automatic and configured to be neither on top nor on bottom.");
    }
  }

  if( !settings->yAxisAuto ){
    if(settings->yAxisLeft && settings->yAxisRight){
      success = false;
      errorMessage->string = toVector(L"y-axis not automatic and configured to be both on top and on bottom.");
    }
    if( !settings->yAxisLeft  &&  !settings->yAxisRight ){
      success = false;
      errorMessage->string = toVector(L"y-axis not automatic and configured to be neither on top nor on bottom.");
    }
  }

  /* Check series lengths. */
  for(i = 0.0; i < settings->scatterPlotSeries->size(); i = i + 1.0){
    series = settings->scatterPlotSeries->at(i);
    if(series->xs->size() != series->ys->size()){
      success = false;
      errorMessage->string = toVector(L"x and y series must be of the same length.");
    }
    if(series->xs->size() == 0.0){
      success = false;
      errorMessage->string = toVector(L"There must be data in the series to be plotted.");
    }
    if(series->linearInterpolation && series->xs->size() == 1.0){
      success = false;
      errorMessage->string = toVector(L"Linear interpolation requires at least two data points to be plotted.");
    }
  }

  /* Check bounds. */
  if( !settings->autoBoundaries ){
    if(settings->xMin >= settings->xMax){
      success = false;
      errorMessage->string = toVector(L"x min is higher than or equal to x max.");
    }
    if(settings->yMin >= settings->yMax){
      success = false;
      errorMessage->string = toVector(L"y min is higher than or equal to y max.");
    }
  }

  /* Check padding. */
  if( !settings->autoPadding ){
    if(2.0*settings->xPadding >= settings->width){
      success = false;
      errorMessage->string = toVector(L"The x padding is more then the width.");
    }
    if(2.0*settings->yPadding >= settings->height){
      success = false;
      errorMessage->string = toVector(L"The y padding is more then the height.");
    }
  }

  /* Check width and height. */
  if(settings->width < 0.0){
    success = false;
    errorMessage->string = toVector(L"The width is less than 0.");
  }
  if(settings->height < 0.0){
    success = false;
    errorMessage->string = toVector(L"The height is less than 0.");
  }

  /* Check point types. */
  for(i = 0.0; i < settings->scatterPlotSeries->size(); i = i + 1.0){
    series = settings->scatterPlotSeries->at(i);

    if(series->lineThickness < 0.0){
      success = false;
      errorMessage->string = toVector(L"The line thickness is less than 0.");
    }

    if( !series->linearInterpolation ){
      /* Point type. */
      found = false;
      if(aStringsEqual(series->pointType, toVector(L"crosses"))){
        found = true;
      }else if(aStringsEqual(series->pointType, toVector(L"circles"))){
        found = true;
      }else if(aStringsEqual(series->pointType, toVector(L"dots"))){
        found = true;
      }else if(aStringsEqual(series->pointType, toVector(L"triangles"))){
        found = true;
      }else if(aStringsEqual(series->pointType, toVector(L"filled triangles"))){
        found = true;
      }else if(aStringsEqual(series->pointType, toVector(L"pixels"))){
        found = true;
      }
      if( !found ){
        success = false;
        errorMessage->string = toVector(L"The point type is unknown.");
      }
    }else{
      /* Line type. */
      found = false;
      if(aStringsEqual(series->lineType, toVector(L"solid"))){
        found = true;
      }else if(aStringsEqual(series->lineType, toVector(L"dashed"))){
        found = true;
      }else if(aStringsEqual(series->lineType, toVector(L"dotted"))){
        found = true;
      }else if(aStringsEqual(series->lineType, toVector(L"dotdash"))){
        found = true;
      }else if(aStringsEqual(series->lineType, toVector(L"longdash"))){
        found = true;
      }else if(aStringsEqual(series->lineType, toVector(L"twodash"))){
        found = true;
      }

      if( !found ){
        success = false;
        errorMessage->string = toVector(L"The line type is unknown.");
      }
    }
  }

  return success;
}
BarPlotSettings *GetDefaultBarPlotSettings(){
  BarPlotSettings *settings;

  settings = new BarPlotSettings();

  settings->width = 800.0;
  settings->height = 600.0;
  settings->autoBoundaries = true;
  settings->yMax = 0.0;
  settings->yMin = 0.0;
  settings->autoPadding = true;
  settings->xPadding = 0.0;
  settings->yPadding = 0.0;
  settings->title = toVector(L"");
  settings->yLabel = toVector(L"");
  settings->barPlotSeries = new vector<BarPlotSeries*> (0.0);
  settings->showGrid = true;
  settings->gridColor = GetGray(0.1);
  settings->autoColor = true;
  settings->grayscaleAutoColor = false;
  settings->autoSpacing = true;
  settings->groupSeparation = 0.0;
  settings->barSeparation = 0.0;
  settings->autoLabels = true;
  settings->xLabels = new vector<StringReference*> (0.0);
  /*settings.autoLabels = false;
        settings.xLabels = new StringReference [5];
        settings.xLabels[0] = CreateStringReference("may 20".toCharArray());
        settings.xLabels[1] = CreateStringReference("jun 20".toCharArray());
        settings.xLabels[2] = CreateStringReference("jul 20".toCharArray());
        settings.xLabels[3] = CreateStringReference("aug 20".toCharArray());
        settings.xLabels[4] = CreateStringReference("sep 20".toCharArray()); */
  settings->barBorder = false;

  return settings;
}
BarPlotSeries *GetDefaultBarPlotSeriesSettings(){
  BarPlotSeries *series;

  series = new BarPlotSeries();

  series->ys = new vector<double> (0.0);
  series->color = GetBlack();

  return series;
}
RGBABitmapImage *DrawBarPlotNoErrorCheck(double width, double height, vector<double> *ys){
  StringReference *errorMessage;
  bool success;
  RGBABitmapImageReference *canvasReference;

  errorMessage = new StringReference();
  canvasReference = CreateRGBABitmapImageReference();

  success = DrawBarPlot(canvasReference, width, height, ys, errorMessage);

  FreeStringReference(errorMessage);

  return canvasReference->image;
}
bool DrawBarPlot(RGBABitmapImageReference *canvasReference, double width, double height, vector<double> *ys, StringReference *errorMessage){
  BarPlotSettings *settings;
  bool success;

  errorMessage = new StringReference();
  settings = GetDefaultBarPlotSettings();

  settings->barPlotSeries = new vector<BarPlotSeries*> (1.0);
  settings->barPlotSeries->at(0) = GetDefaultBarPlotSeriesSettings();
  delete settings->barPlotSeries->at(0)->ys;
  settings->barPlotSeries->at(0)->ys = ys;
  settings->width = width;
  settings->height = height;

  success = DrawBarPlotFromSettings(canvasReference, settings, errorMessage);

  return success;
}
bool DrawBarPlotFromSettings(RGBABitmapImageReference *canvasReference, BarPlotSettings *settings, StringReference *errorMessage){
  double xPadding, yPadding;
  double xPixelMin, yPixelMin, yPixelMax, xPixelMax;
  double xLengthPixels, yLengthPixels;
  double s, n, y, x, w, h, yMin, yMax, b, i, py, yValue;
  vector<RGBA*> *colors;
  vector<double> *ys, *yGridPositions;
  double yTop, yBottom, ss, bs;
  double groupSeparation, barSeparation, barWidth, textwidth;
  StringArrayReference *yLabels;
  NumberArrayReference *yLabelPriorities;
  vector<Rectangle*> *occupied;
  NumberReference *nextRectangle;
  RGBA *gridLabelColor, *barColor;
  vector<wchar_t> *label;
  bool success;
  RGBABitmapImage *canvas;

  success = BarPlotSettingsIsValid(settings, errorMessage);

  if(success){
    canvas = CreateImage(settings->width, settings->height, GetWhite());

    ss = settings->barPlotSeries->size();
    gridLabelColor = GetGray(0.5);

    /* padding */
    if(settings->autoPadding){
      xPadding = floor(GetDefaultPaddingPercentage()*ImageWidth(canvas));
      yPadding = floor(GetDefaultPaddingPercentage()*ImageHeight(canvas));
    }else{
      xPadding = settings->xPadding;
      yPadding = settings->yPadding;
    }

    /* Draw title */
    DrawText(canvas, floor(ImageWidth(canvas)/2.0 - GetTextWidth(settings->title)/2.0), floor(yPadding/3.0), settings->title, GetBlack());
    DrawTextUpwards(canvas, 10.0, floor(ImageHeight(canvas)/2.0 - GetTextWidth(settings->yLabel)/2.0), settings->yLabel, GetBlack());

    /* min and max */
    if(settings->autoBoundaries){
      if(ss >= 1.0){
        yMax = GetMaximum(settings->barPlotSeries->at(0)->ys);
        yMin = fmin(0.0, GetMinimum(settings->barPlotSeries->at(0)->ys));

        for(s = 0.0; s < ss; s = s + 1.0){
          yMax = fmax(yMax, GetMaximum(settings->barPlotSeries->at(s)->ys));
          yMin = fmin(yMin, GetMinimum(settings->barPlotSeries->at(s)->ys));
        }
      }else{
        yMax = 10.0;
        yMin = 0.0;
      }
    }else{
      yMin = settings->yMin;
      yMax = settings->yMax;
    }

    /* boundaries */
    xPixelMin = xPadding;
    yPixelMin = yPadding;
    xPixelMax = ImageWidth(canvas) - xPadding;
    yPixelMax = ImageHeight(canvas) - yPadding;
    xLengthPixels = xPixelMax - xPixelMin;
    yLengthPixels = yPixelMax - yPixelMin;

    /* Draw boundary. */
    DrawRectangle1px(canvas, xPixelMin, yPixelMin, xLengthPixels, yLengthPixels, settings->gridColor);

    /* Draw grid lines. */
    yLabels = new StringArrayReference();
    yLabelPriorities = new NumberArrayReference();
    yGridPositions = ComputeGridLinePositions(yMin, yMax, yLabels, yLabelPriorities);

    if(settings->showGrid){
      /* Y-grid */
      for(i = 0.0; i < yGridPositions->size(); i = i + 1.0){
        y = yGridPositions->at(i);
        py = MapYCoordinate(y, yMin, yMax, yPixelMin, yPixelMax);
        DrawLine1px(canvas, xPixelMin, py, xPixelMax, py, settings->gridColor);
      }
    }

    /* Draw origin. */
    if(yMin < 0.0 && yMax > 0.0){
      py = MapYCoordinate(0.0, yMin, yMax, yPixelMin, yPixelMax);
      DrawLine1px(canvas, xPixelMin, py, xPixelMax, py, settings->gridColor);
    }

    /* Labels */
    occupied = new vector<Rectangle*> (yLabels->stringArray->size());
    for(i = 0.0; i < occupied->size(); i = i + 1.0){
      occupied->at(i) = CreateRectangle(0.0, 0.0, 0.0, 0.0);
    }
    nextRectangle = CreateNumberReference(0.0);

    for(i = 1.0; i <= 5.0; i = i + 1.0){
      DrawYLabelsForPriority(i, yMin, xPixelMin, yMax, yPixelMin, yPixelMax, nextRectangle, gridLabelColor, canvas, yGridPositions, yLabels, yLabelPriorities, occupied, true);
    }

    /* Draw bars. */
    if(settings->autoColor){
      if( !settings->grayscaleAutoColor ){
        colors = Get8HighContrastColors();
      }else{
        colors = new vector<RGBA*> (ss);
        if(ss > 1.0){
          for(i = 0.0; i < ss; i = i + 1.0){
            colors->at(i) = GetGray(0.7 - (i/ss)*0.7);
          }
        }else{
          colors->at(0) = GetGray(0.5);
        }
      }
    }else{
      colors = new vector<RGBA*> (0.0);
    }

    /* distances */
    bs = settings->barPlotSeries->at(0)->ys->size();

    if(settings->autoSpacing){
      groupSeparation = ImageWidth(canvas)*0.05;
      barSeparation = ImageWidth(canvas)*0.005;
    }else{
      groupSeparation = settings->groupSeparation;
      barSeparation = settings->barSeparation;
    }

    barWidth = (xLengthPixels - groupSeparation*(bs - 1.0) - barSeparation*(bs*(ss - 1.0)))/(bs*ss);

    /* Draw bars. */
    b = 0.0;
    for(n = 0.0; n < bs; n = n + 1.0){
      for(s = 0.0; s < ss; s = s + 1.0){
        ys = settings->barPlotSeries->at(s)->ys;

        yValue = ys->at(n);

        yBottom = MapYCoordinate(yValue, yMin, yMax, yPixelMin, yPixelMax);
        yTop = MapYCoordinate(0.0, yMin, yMax, yPixelMin, yPixelMax);

        x = xPixelMin + n*(groupSeparation + ss*barWidth) + s*(barWidth) + b*barSeparation;
        w = barWidth;

        if(yValue >= 0.0){
          y = yBottom;
          h = yTop - y;
        }else{
          y = yTop;
          h = yBottom - yTop;
        }

        /* Cut at boundaries. */
        if(y < yPixelMin && y + h > yPixelMax){
          y = yPixelMin;
          h = yPixelMax - yPixelMin;
        }else if(y < yPixelMin){
          y = yPixelMin;
          if(yValue >= 0.0){
            h = yTop - y;
          }else{
            h = yBottom - y;
          }
        }else if(y + h > yPixelMax){
          h = yPixelMax - y;
        }

        /* Get color */
        if(settings->autoColor){
          barColor = colors->at(s);
        }else{
          barColor = settings->barPlotSeries->at(s)->color;
        }

        /* Draw */
        if(settings->barBorder){
          DrawFilledRectangleWithBorder(canvas, Round(x), Round(y), Round(w), Round(h), GetBlack(), barColor);
        }else{
          DrawFilledRectangle(canvas, Round(x), Round(y), Round(w), Round(h), barColor);
        }

        b = b + 1.0;
      }
      b = b - 1.0;
    }

    /* x-labels */
    for(n = 0.0; n < bs; n = n + 1.0){
      if(settings->autoLabels){
        label = CreateStringDecimalFromNumber(n + 1.0);
      }else{
        label = settings->xLabels->at(n)->string;
      }

      textwidth = GetTextWidth(label);

      x = xPixelMin + (n + 0.5)*(ss*barWidth + (ss - 1.0)*barSeparation) + n*groupSeparation - textwidth/2.0;

      DrawText(canvas, floor(x), ImageHeight(canvas) - yPadding + 20.0, label, gridLabelColor);

      b = b + 1.0;
    }

    canvasReference->image = canvas;
  }

  return success;
}
bool BarPlotSettingsIsValid(BarPlotSettings *settings, StringReference *errorMessage){
  bool success, lengthSet;
  BarPlotSeries *series;
  double i, length;

  success = true;

  /* Check series lengths. */
  lengthSet = false;
  length = 0.0;
  for(i = 0.0; i < settings->barPlotSeries->size(); i = i + 1.0){
    series = settings->barPlotSeries->at(i);

    if( !lengthSet ){
      length = series->ys->size();
      lengthSet = true;
    }else if(length != series->ys->size()){
      success = false;
      errorMessage->string = toVector(L"The number of data points must be equal for all series.");
    }
  }

  /* Check bounds. */
  if( !settings->autoBoundaries ){
    if(settings->yMin >= settings->yMax){
      success = false;
      errorMessage->string = toVector(L"Minimum y lower than maximum y.");
    }
  }

  /* Check padding. */
  if( !settings->autoPadding ){
    if(2.0*settings->xPadding >= settings->width){
      success = false;
      errorMessage->string = toVector(L"Double the horizontal padding is larger than or equal to the width.");
    }
    if(2.0*settings->yPadding >= settings->height){
      success = false;
      errorMessage->string = toVector(L"Double the vertical padding is larger than or equal to the height.");
    }
  }

  /* Check width and height. */
  if(settings->width < 0.0){
    success = false;
    errorMessage->string = toVector(L"Width lower than zero.");
  }
  if(settings->height < 0.0){
    success = false;
    errorMessage->string = toVector(L"Height lower than zero.");
  }

  /* Check spacing */
  if( !settings->autoSpacing ){
    if(settings->groupSeparation < 0.0){
      success = false;
      errorMessage->string = toVector(L"Group separation lower than zero.");
    }
    if(settings->barSeparation < 0.0){
      success = false;
      errorMessage->string = toVector(L"Bar separation lower than zero.");
    }
  }

  return success;
}
double GetMinimum(vector<double> *data){
  double i, minimum;

  minimum = data->at(0);
  for(i = 0.0; i < data->size(); i = i + 1.0){
    minimum = fmin(minimum, data->at(i));
  }

  return minimum;
}
double GetMaximum(vector<double> *data){
  double i, maximum;

  maximum = data->at(0);
  for(i = 0.0; i < data->size(); i = i + 1.0){
    maximum = fmax(maximum, data->at(i));
  }

  return maximum;
}
double RoundToDigits(double element, double digitsAfterPoint){
  return Round(element*pow(10.0, digitsAfterPoint))/pow(10.0, digitsAfterPoint);
}
double test(){
  double z;
  vector<double> *gridlines;
  NumberReference *failures;
  StringArrayReference *labels;
  NumberArrayReference *labelPriorities;
  RGBABitmapImageReference *imageReference;
  vector<double> *xs, *ys;
  StringReference *errorMessage;
  bool success;

  failures = CreateNumberReference(0.0);
  errorMessage = CreateStringReference(toVector(L""));

  imageReference = CreateRGBABitmapImageReference();

  labels = new StringArrayReference();
  labelPriorities = new NumberArrayReference();

  z = 10.0;
  gridlines = ComputeGridLinePositions( -z/2.0, z/2.0, labels, labelPriorities);
  AssertEquals(gridlines->size(), 11.0, failures);

  z = 9.0;
  gridlines = ComputeGridLinePositions( -z/2.0, z/2.0, labels, labelPriorities);
  AssertEquals(gridlines->size(), 19.0, failures);

  z = 8.0;
  gridlines = ComputeGridLinePositions( -z/2.0, z/2.0, labels, labelPriorities);
  AssertEquals(gridlines->size(), 17.0, failures);

  z = 7.0;
  gridlines = ComputeGridLinePositions( -z/2.0, z/2.0, labels, labelPriorities);
  AssertEquals(gridlines->size(), 15.0, failures);

  z = 6.0;
  gridlines = ComputeGridLinePositions( -z/2.0, z/2.0, labels, labelPriorities);
  AssertEquals(gridlines->size(), 13.0, failures);

  z = 5.0;
  gridlines = ComputeGridLinePositions( -z/2.0, z/2.0, labels, labelPriorities);
  AssertEquals(gridlines->size(), 21.0, failures);

  z = 4.0;
  gridlines = ComputeGridLinePositions( -z/2.0, z/2.0, labels, labelPriorities);
  AssertEquals(gridlines->size(), 17.0, failures);

  z = 3.0;
  gridlines = ComputeGridLinePositions( -z/2.0, z/2.0, labels, labelPriorities);
  AssertEquals(gridlines->size(), 31.0, failures);

  z = 2.0;
  gridlines = ComputeGridLinePositions( -z/2.0, z/2.0, labels, labelPriorities);
  AssertEquals(gridlines->size(), 21.0, failures);

  xs = new vector<double> (5.0);
  xs->at(0) =  -2.0;
  xs->at(1) =  -1.0;
  xs->at(2) = 0.0;
  xs->at(3) = 1.0;
  xs->at(4) = 2.0;
  ys = new vector<double> (5.0);
  ys->at(0) = 2.0;
  ys->at(1) =  -1.0;
  ys->at(2) =  -2.0;
  ys->at(3) =  -1.0;
  ys->at(4) = 2.0;
  success = DrawScatterPlot(imageReference, 800.0, 600.0, xs, ys, errorMessage);

  AssertTrue(success, failures);

  if(success){
    success = DrawBarPlot(imageReference, 800.0, 600.0, ys, errorMessage);

    AssertTrue(success, failures);

    if(success){
      TestMapping(failures);
      TestMapping2(failures);
    }
  }

  return failures->numberValue;
}
void TestMapping(NumberReference *failures){
  ScatterPlotSeries *series;
  ScatterPlotSettings *settings;
  RGBABitmapImageReference *imageReference;
  double x1, y1;
  StringReference *errorMessage;
  bool success;

  errorMessage = CreateStringReference(toVector(L""));

  series = GetDefaultScatterPlotSeriesSettings();

  series->xs = new vector<double> (5.0);
  series->xs->at(0) = -2.0;
  series->xs->at(1) = -1.0;
  series->xs->at(2) = 0.0;
  series->xs->at(3) = 1.0;
  series->xs->at(4) = 2.0;
  series->ys = new vector<double> (5.0);
  series->ys->at(0) = -2.0;
  series->ys->at(1) = -1.0;
  series->ys->at(2) = -2.0;
  series->ys->at(3) = -1.0;
  series->ys->at(4) = 2.0;
  series->linearInterpolation = true;
  series->lineType = toVector(L"dashed");
  series->lineThickness = 2.0;
  series->color = GetGray(0.3);

  settings = GetDefaultScatterPlotSettings();
  settings->width = 600.0;
  settings->height = 400.0;
  settings->autoBoundaries = true;
  settings->autoPadding = true;
  settings->title = toVector(L"x^2 - 2");
  settings->xLabel = toVector(L"X axis");
  settings->yLabel = toVector(L"Y axis");
  settings->scatterPlotSeries = new vector<ScatterPlotSeries*> (1.0);
  settings->scatterPlotSeries->at(0) = series;

  imageReference = CreateRGBABitmapImageReference();
  success = DrawScatterPlotFromSettings(imageReference, settings, errorMessage);

  AssertTrue(success, failures);

  if(success){
    x1 = MapXCoordinateAutoSettings( -1.0, imageReference->image, series->xs);
    y1 = MapYCoordinateAutoSettings( -1.0, imageReference->image, series->ys);

    AssertEquals(x1, 180.0, failures);
    AssertEquals(y1, 280.0, failures);
  }
}
void TestMapping2(NumberReference *failures){
  vector<double> *xs, *ys, *xs2, *ys2;
  double i, x, y, w, h, xMin, xMax, yMin, yMax;
  RGBABitmapImageReference *canvasReference;
  ScatterPlotSettings *settings;
  double points;
  double x1, y1;
  StringReference *errorMessage;
  bool success;

  errorMessage = CreateStringReference(toVector(L""));

  points = 300.0;
  w = 600.0*2.0;
  h = 300.0*2.0;
  xMin = 0.0;
  xMax = 150.0;
  yMin = 0.0;
  yMax = 1.0;

  xs = new vector<double> (points);
  ys = new vector<double> (points);
  xs2 = new vector<double> (points);
  ys2 = new vector<double> (points);

  for(i = 0.0; i < points; i = i + 1.0){
    x = xMin + (xMax - xMin)/(points - 1.0)*i;
    /* points - 1d is to ensure both extremeties are included. */
    y = x/(x + 7.0);

    xs->at(i) = x;
    ys->at(i) = y;

    y = 1.4*x/(x + 7.0)*(1.0 - (atan((x/1.5 - 30.0)/5.0)/1.6 + 1.0)/2.0);

    xs2->at(i) = x;
    ys2->at(i) = y;
  }

  settings = GetDefaultScatterPlotSettings();

  settings->scatterPlotSeries = new vector<ScatterPlotSeries*> (2.0);
  settings->scatterPlotSeries->at(0) = new ScatterPlotSeries();
  settings->scatterPlotSeries->at(0)->xs = xs;
  settings->scatterPlotSeries->at(0)->ys = ys;
  settings->scatterPlotSeries->at(0)->linearInterpolation = true;
  settings->scatterPlotSeries->at(0)->lineType = toVector(L"solid");
  settings->scatterPlotSeries->at(0)->lineThickness = 3.0;
  settings->scatterPlotSeries->at(0)->color = CreateRGBColor(1.0, 0.0, 0.0);
  settings->scatterPlotSeries->at(1) = new ScatterPlotSeries();
  settings->scatterPlotSeries->at(1)->xs = xs2;
  settings->scatterPlotSeries->at(1)->ys = ys2;
  settings->scatterPlotSeries->at(1)->linearInterpolation = true;
  settings->scatterPlotSeries->at(1)->lineType = toVector(L"solid");
  settings->scatterPlotSeries->at(1)->lineThickness = 3.0;
  settings->scatterPlotSeries->at(1)->color = CreateRGBColor(0.0, 0.0, 1.0);
  settings->autoBoundaries = false;
  settings->xMin = xMin;
  settings->xMax = xMax;
  settings->yMin = yMin;
  settings->yMax = yMax;
  settings->yLabel = toVector(L"");
  settings->xLabel = toVector(L"Features");
  settings->title = toVector(L"");
  settings->width = w;
  settings->height = h;

  canvasReference = CreateRGBABitmapImageReference();

  success = DrawScatterPlotFromSettings(canvasReference, settings, errorMessage);

  AssertTrue(success, failures);

  if(success){
    x1 = MapXCoordinateBasedOnSettings(27.0, settings);
    y1 = MapYCoordinateBasedOnSettings(1.0, settings);

    AssertEquals(floor(x1), 292.0, failures);
    AssertEquals(y1, 60.0, failures);
  }
}
void ExampleRegression(RGBABitmapImageReference *image){
  vector<wchar_t> *xsStr, *ysStr;
  vector<double> *xs, *ys, *xs2, *ys2;
  StringReference *errorMessage;
  bool success;
  ScatterPlotSettings *settings;

  errorMessage = CreateStringReference(toVector(L""));

  xsStr = toVector(L"20.1, 7.1, 16.1, 14.9, 16.7, 8.8, 9.7, 10.3, 22, 16.2, 12.1, 10.3, 14.5, 12.4, 9.6, 12.2, 10.8, 14.7, 19.7, 11.2, 10.1, 11, 12.2, 9.2, 23.5, 9.4, 15.3, 9.6, 11.1, 5.3, 7.8, 25.3, 16.5, 12.6, 12, 11.5, 17.1, 11.2, 12.2, 10.6, 19.9, 14.5, 15.5, 17.4, 8.4, 10.3, 10.2, 12.5, 16.7, 8.5, 12.2");
  ysStr = toVector(L"31.5, 18.9, 35, 31.6, 22.6, 26.2, 14.1, 24.7, 44.8, 23.2, 31.4, 17.7, 18.4, 23.4, 22.6, 16.4, 21.4, 26.5, 31.7, 11.9, 20, 12.5, 18, 14.2, 37.6, 22.2, 17.8, 18.3, 28, 8.1, 14.7, 37.8, 15.7, 28.6, 11.7, 20.1, 30.1, 18.2, 17.2, 19.6, 29.2, 17.3, 28.2, 38.2, 17.8, 10.4, 19, 16.8, 21.5, 15.9, 17.7");

  xs = StringToNumberArray(xsStr);
  ys = StringToNumberArray(ysStr);

  settings = GetDefaultScatterPlotSettings();

  settings->scatterPlotSeries = new vector<ScatterPlotSeries*> (2.0);
  settings->scatterPlotSeries->at(0) = new ScatterPlotSeries();
  settings->scatterPlotSeries->at(0)->xs = xs;
  settings->scatterPlotSeries->at(0)->ys = ys;
  settings->scatterPlotSeries->at(0)->linearInterpolation = false;
  settings->scatterPlotSeries->at(0)->pointType = toVector(L"dots");
  settings->scatterPlotSeries->at(0)->color = CreateRGBColor(1.0, 0.0, 0.0);

  /*OrdinaryLeastSquaresWithIntercept(); */
  xs2 = new vector<double> (2.0);
  ys2 = new vector<double> (2.0);

  xs2->at(0) = 5.0;
  ys2->at(0) = 12.0;
  xs2->at(1) = 25.0;
  ys2->at(1) = 39.0;

  settings->scatterPlotSeries->at(1) = new ScatterPlotSeries();
  settings->scatterPlotSeries->at(1)->xs = xs2;
  settings->scatterPlotSeries->at(1)->ys = ys2;
  settings->scatterPlotSeries->at(1)->linearInterpolation = true;
  settings->scatterPlotSeries->at(1)->lineType = toVector(L"solid");
  settings->scatterPlotSeries->at(1)->lineThickness = 2.0;
  settings->scatterPlotSeries->at(1)->color = CreateRGBColor(0.0, 0.0, 1.0);

  settings->autoBoundaries = true;
  settings->yLabel = toVector(L"");
  settings->xLabel = toVector(L"");
  settings->title = toVector(L"");
  settings->width = 600.0;
  settings->height = 400.0;

  success = DrawScatterPlotFromSettings(image, settings, errorMessage);
}
void BarPlotExample(RGBABitmapImageReference *imageReference){
  vector<double> *ys1, *ys2, *ys3;
  BarPlotSettings *settings;
  StringReference *errorMessage;
  bool success;

  errorMessage = new StringReference();

  ys1 = StringToNumberArray(toVector(L"1, 2, 3, 4, 5"));
  ys2 = StringToNumberArray(toVector(L"5, 4, 3, 2, 1"));
  ys3 = StringToNumberArray(toVector(L"10, 2, 4, 3, 4"));

  settings = GetDefaultBarPlotSettings();

  settings->autoBoundaries = true;
  /*settings.yMax; */
  /*settings.yMin; */
  settings->autoPadding = true;
  /*settings.xPadding; */
  /*settings.yPadding; */
  settings->title = toVector(L"title");
  settings->showGrid = true;
  settings->gridColor = GetGray(0.1);
  settings->yLabel = toVector(L"y label");
  settings->autoColor = true;
  settings->grayscaleAutoColor = false;
  settings->autoSpacing = true;
  /*settings.groupSeparation; */
  /*settings.barSeparation; */
  settings->autoLabels = false;
  settings->xLabels = new vector<StringReference*> (5.0);
  settings->xLabels->at(0) = CreateStringReference(toVector(L"may 20"));
  settings->xLabels->at(1) = CreateStringReference(toVector(L"jun 20"));
  settings->xLabels->at(2) = CreateStringReference(toVector(L"jul 20"));
  settings->xLabels->at(3) = CreateStringReference(toVector(L"aug 20"));
  settings->xLabels->at(4) = CreateStringReference(toVector(L"sep 20"));
  /*settings.colors; */
  settings->barBorder = true;

  settings->barPlotSeries = new vector<BarPlotSeries*> (3.0);
  settings->barPlotSeries->at(0) = GetDefaultBarPlotSeriesSettings();
  settings->barPlotSeries->at(0)->ys = ys1;
  settings->barPlotSeries->at(1) = GetDefaultBarPlotSeriesSettings();
  settings->barPlotSeries->at(1)->ys = ys2;
  settings->barPlotSeries->at(2) = GetDefaultBarPlotSeriesSettings();
  settings->barPlotSeries->at(2)->ys = ys3;

  success = DrawBarPlotFromSettings(imageReference, settings, errorMessage);
}
RGBA *GetBlack(){
  RGBA *black;
  black = new RGBA();
  black->a = 1.0;
  black->r = 0.0;
  black->g = 0.0;
  black->b = 0.0;
  return black;
}
RGBA *GetWhite(){
  RGBA *white;
  white = new RGBA();
  white->a = 1.0;
  white->r = 1.0;
  white->g = 1.0;
  white->b = 1.0;
  return white;
}
RGBA *GetTransparent(){
  RGBA *transparent;
  transparent = new RGBA();
  transparent->a = 0.0;
  transparent->r = 0.0;
  transparent->g = 0.0;
  transparent->b = 0.0;
  return transparent;
}
RGBA *GetGray(double percentage){
  RGBA *black;
  black = new RGBA();
  black->a = 1.0;
  black->r = 1.0 - percentage;
  black->g = 1.0 - percentage;
  black->b = 1.0 - percentage;
  return black;
}
RGBA *CreateRGBColor(double r, double g, double b){
  RGBA *color;
  color = new RGBA();
  color->a = 1.0;
  color->r = r;
  color->g = g;
  color->b = b;
  return color;
}
RGBA *CreateRGBAColor(double r, double g, double b, double a){
  RGBA *color;
  color = new RGBA();
  color->a = a;
  color->r = r;
  color->g = g;
  color->b = b;
  return color;
}
RGBABitmapImage *CreateImage(double w, double h, RGBA *color){
  RGBABitmapImage *image;
  double i, j;

  image = new RGBABitmapImage();
  image->x = new vector<RGBABitmap*> (w);
  for(i = 0.0; i < w; i = i + 1.0){
    image->x->at(i) = new RGBABitmap();
    image->x->at(i)->y = new vector<RGBA*> (h);
    for(j = 0.0; j < h; j = j + 1.0){
      image->x->at(i)->y->at(j) = new RGBA();
      SetPixel(image, i, j, color);
    }
  }

  return image;
}
void DeleteImage(RGBABitmapImage *image){
  double i, j, w, h;

  w = ImageWidth(image);
  h = ImageHeight(image);

  for(i = 0.0; i < w; i = i + 1.0){
    for(j = 0.0; j < h; j = j + 1.0){
      delete image->x->at(i)->y->at(j);
    }
    delete image->x->at(i);
  }
  delete image;
}
double ImageWidth(RGBABitmapImage *image){
  return image->x->size();
}
double ImageHeight(RGBABitmapImage *image){
  double height;

  if(ImageWidth(image) == 0.0){
    height = 0.0;
  }else{
    height = image->x->at(0)->y->size();
  }

  return height;
}
void SetPixel(RGBABitmapImage *image, double x, double y, RGBA *color){
  if(x >= 0.0 && x < ImageWidth(image) && y >= 0.0 && y < ImageHeight(image)){
    image->x->at(x)->y->at(y)->a = color->a;
    image->x->at(x)->y->at(y)->r = color->r;
    image->x->at(x)->y->at(y)->g = color->g;
    image->x->at(x)->y->at(y)->b = color->b;
  }
}
void DrawPixel(RGBABitmapImage *image, double x, double y, RGBA *color){
  double ra, ga, ba, aa;
  double rb, gb, bb, ab;
  double ro, go, bo, ao;

  if(x >= 0.0 && x < ImageWidth(image) && y >= 0.0 && y < ImageHeight(image)){
    ra = color->r;
    ga = color->g;
    ba = color->b;
    aa = color->a;

    rb = image->x->at(x)->y->at(y)->r;
    gb = image->x->at(x)->y->at(y)->g;
    bb = image->x->at(x)->y->at(y)->b;
    ab = image->x->at(x)->y->at(y)->a;

    ao = CombineAlpha(aa, ab);

    ro = AlphaBlend(ra, aa, rb, ab, ao);
    go = AlphaBlend(ga, aa, gb, ab, ao);
    bo = AlphaBlend(ba, aa, bb, ab, ao);

    image->x->at(x)->y->at(y)->r = ro;
    image->x->at(x)->y->at(y)->g = go;
    image->x->at(x)->y->at(y)->b = bo;
    image->x->at(x)->y->at(y)->a = ao;
  }
}
double CombineAlpha(double as, double ad){
  return as + ad*(1.0 - as);
}
double AlphaBlend(double cs, double as, double cd, double ad, double ao){
  return (cs*as + cd*ad*(1.0 - as))/ao;
}
void DrawHorizontalLine1px(RGBABitmapImage *image, double x, double y, double length, RGBA *color){
  double i;

  for(i = 0.0; i < length; i = i + 1.0){
    DrawPixel(image, x + i, y, color);
  }
}
void DrawVerticalLine1px(RGBABitmapImage *image, double x, double y, double height, RGBA *color){
  double i;

  for(i = 0.0; i < height; i = i + 1.0){
    DrawPixel(image, x, y + i, color);
  }
}
void DrawRectangle1px(RGBABitmapImage *image, double x, double y, double width, double height, RGBA *color){
  DrawHorizontalLine1px(image, x, y, width + 1.0, color);
  DrawVerticalLine1px(image, x, y + 1.0, height + 1.0 - 1.0, color);
  DrawVerticalLine1px(image, x + width, y + 1.0, height + 1.0 - 1.0, color);
  DrawHorizontalLine1px(image, x + 1.0, y + height, width + 1.0 - 2.0, color);
}
void DrawImageOnImage(RGBABitmapImage *dst, RGBABitmapImage *src, double topx, double topy){
  double y, x;

  for(y = 0.0; y < ImageHeight(src); y = y + 1.0){
    for(x = 0.0; x < ImageWidth(src); x = x + 1.0){
      if(topx + x >= 0.0 && topx + x < ImageWidth(dst) && topy + y >= 0.0 && topy + y < ImageHeight(dst)){
        DrawPixel(dst, topx + x, topy + y, src->x->at(x)->y->at(y));
      }
    }
  }
}
void DrawLine1px(RGBABitmapImage *image, double x0, double y0, double x1, double y1, RGBA *color){
  XiaolinWusLineAlgorithm(image, x0, y0, x1, y1, color);
}
void XiaolinWusLineAlgorithm(RGBABitmapImage *image, double x0, double y0, double x1, double y1, RGBA *color){
  bool steep;
  double x, t, dx, dy, g, xEnd, yEnd, xGap, xpxl1, ypxl1, intery, xpxl2, ypxl2, olda;

  olda = color->a;

  steep = abs(y1 - y0) > abs(x1 - x0);

  if(steep){
    t = x0;
    x0 = y0;
    y0 = t;

    t = x1;
    x1 = y1;
    y1 = t;
  }
  if(x0 > x1){
    t = x0;
    x0 = x1;
    x1 = t;

    t = y0;
    y0 = y1;
    y1 = t;
  }

  dx = x1 - x0;
  dy = y1 - y0;
  g = dy/dx;

  if(dx == 0.0){
    g = 1.0;
  }

  xEnd = Round(x0);
  yEnd = y0 + g*(xEnd - x0);
  xGap = OneMinusFractionalPart(x0 + 0.5);
  xpxl1 = xEnd;
  ypxl1 = floor(yEnd);
  if(steep){
    DrawPixel(image, ypxl1, xpxl1, SetBrightness(color, OneMinusFractionalPart(yEnd)*xGap));
    DrawPixel(image, ypxl1 + 1.0, xpxl1, SetBrightness(color, FractionalPart(yEnd)*xGap));
  }else{
    DrawPixel(image, xpxl1, ypxl1, SetBrightness(color, OneMinusFractionalPart(yEnd)*xGap));
    DrawPixel(image, xpxl1, ypxl1 + 1.0, SetBrightness(color, FractionalPart(yEnd)*xGap));
  }
  intery = yEnd + g;

  xEnd = Round(x1);
  yEnd = y1 + g*(xEnd - x1);
  xGap = FractionalPart(x1 + 0.5);
  xpxl2 = xEnd;
  ypxl2 = floor(yEnd);
  if(steep){
    DrawPixel(image, ypxl2, xpxl2, SetBrightness(color, OneMinusFractionalPart(yEnd)*xGap));
    DrawPixel(image, ypxl2 + 1.0, xpxl2, SetBrightness(color, FractionalPart(yEnd)*xGap));
  }else{
    DrawPixel(image, xpxl2, ypxl2, SetBrightness(color, OneMinusFractionalPart(yEnd)*xGap));
    DrawPixel(image, xpxl2, ypxl2 + 1.0, SetBrightness(color, FractionalPart(yEnd)*xGap));
  }

  if(steep){
    for(x = xpxl1 + 1.0; x <= xpxl2 - 1.0; x = x + 1.0){
      DrawPixel(image, floor(intery), x, SetBrightness(color, OneMinusFractionalPart(intery)));
      DrawPixel(image, floor(intery) + 1.0, x, SetBrightness(color, FractionalPart(intery)));
      intery = intery + g;
    }
  }else{
    for(x = xpxl1 + 1.0; x <= xpxl2 - 1.0; x = x + 1.0){
      DrawPixel(image, x, floor(intery), SetBrightness(color, OneMinusFractionalPart(intery)));
      DrawPixel(image, x, floor(intery) + 1.0, SetBrightness(color, FractionalPart(intery)));
      intery = intery + g;
    }
  }

  color->a = olda;
}
double OneMinusFractionalPart(double x){
  return 1.0 - FractionalPart(x);
}
double FractionalPart(double x){
  return x - floor(x);
}
RGBA *SetBrightness(RGBA *color, double newBrightness){
  color->a = newBrightness;
  return color;
}
void DrawQuadraticBezierCurve(RGBABitmapImage *image, double x0, double y0, double cx, double cy, double x1, double y1, RGBA *color){
  double t, dt, dx, dy;
  NumberReference *xs, *ys, *xe, *ye;

  dx = abs(x0 - x1);
  dy = abs(y0 - y1);

  dt = 1.0/sqrt(pow(dx, 2.0) + pow(dy, 2.0));

  xs = new NumberReference();
  ys = new NumberReference();
  xe = new NumberReference();
  ye = new NumberReference();

  QuadraticBezierPoint(x0, y0, cx, cy, x1, y1, 0.0, xs, ys);
  for(t = dt; t <= 1.0; t = t + dt){
    QuadraticBezierPoint(x0, y0, cx, cy, x1, y1, t, xe, ye);
    DrawLine1px(image, xs->numberValue, ys->numberValue, xe->numberValue, ye->numberValue, color);
    xs->numberValue = xe->numberValue;
    ys->numberValue = ye->numberValue;
  }

  delete xs;
  delete ys;
  delete xe;
  delete ye;
}
void QuadraticBezierPoint(double x0, double y0, double cx, double cy, double x1, double y1, double t, NumberReference *x, NumberReference *y){
  x->numberValue = pow(1.0 - t, 2.0)*x0 + (1.0 - t)*2.0*t*cx + pow(t, 2.0)*x1;
  y->numberValue = pow(1.0 - t, 2.0)*y0 + (1.0 - t)*2.0*t*cy + pow(t, 2.0)*y1;
}
void DrawCubicBezierCurve(RGBABitmapImage *image, double x0, double y0, double c0x, double c0y, double c1x, double c1y, double x1, double y1, RGBA *color){
  double t, dt, dx, dy;
  NumberReference *xs, *ys, *xe, *ye;

  dx = abs(x0 - x1);
  dy = abs(y0 - y1);

  dt = 1.0/sqrt(pow(dx, 2.0) + pow(dy, 2.0));

  xs = new NumberReference();
  ys = new NumberReference();
  xe = new NumberReference();
  ye = new NumberReference();

  CubicBezierPoint(x0, y0, c0x, c0y, c1x, c1y, x1, y1, 0.0, xs, ys);
  for(t = dt; t <= 1.0; t = t + dt){
    CubicBezierPoint(x0, y0, c0x, c0y, c1x, c1y, x1, y1, t, xe, ye);
    DrawLine1px(image, xs->numberValue, ys->numberValue, xe->numberValue, ye->numberValue, color);
    xs->numberValue = xe->numberValue;
    ys->numberValue = ye->numberValue;
  }

  delete xs;
  delete ys;
  delete xe;
  delete ye;
}
void CubicBezierPoint(double x0, double y0, double c0x, double c0y, double c1x, double c1y, double x1, double y1, double t, NumberReference *x, NumberReference *y){
  x->numberValue = pow(1.0 - t, 3.0)*x0 + pow(1.0 - t, 2.0)*3.0*t*c0x + (1.0 - t)*3.0*pow(t, 2.0)*c1x + pow(t, 3.0)*x1;

  y->numberValue = pow(1.0 - t, 3.0)*y0 + pow(1.0 - t, 2.0)*3.0*t*c0y + (1.0 - t)*3.0*pow(t, 2.0)*c1y + pow(t, 3.0)*y1;
}
RGBABitmapImage *CopyImage(RGBABitmapImage *image){
  RGBABitmapImage *copy;
  double i, j;

  copy = CreateImage(ImageWidth(image), ImageHeight(image), GetTransparent());

  for(i = 0.0; i < ImageWidth(image); i = i + 1.0){
    for(j = 0.0; j < ImageHeight(image); j = j + 1.0){
      SetPixel(copy, i, j, image->x->at(i)->y->at(j));
    }
  }

  return copy;
}
RGBA *GetImagePixel(RGBABitmapImage *image, double x, double y){
  return image->x->at(x)->y->at(y);
}
void HorizontalFlip(RGBABitmapImage *img){
  double y, x;
  double tmp;
  RGBA *c1, *c2;

  for(y = 0.0; y < ImageHeight(img); y = y + 1.0){
    for(x = 0.0; x < ImageWidth(img)/2.0; x = x + 1.0){
      c1 = img->x->at(x)->y->at(y);
      c2 = img->x->at(ImageWidth(img) - 1.0 - x)->y->at(y);

      tmp = c1->a;
      c1->a = c2->a;
      c2->a = tmp;

      tmp = c1->r;
      c1->r = c2->r;
      c2->r = tmp;

      tmp = c1->g;
      c1->g = c2->g;
      c2->g = tmp;

      tmp = c1->b;
      c1->b = c2->b;
      c2->b = tmp;
    }
  }
}
void DrawFilledRectangle(RGBABitmapImage *image, double x, double y, double w, double h, RGBA *color){
  double i, j;

  for(i = 0.0; i < w; i = i + 1.0){
    for(j = 0.0; j < h; j = j + 1.0){
      SetPixel(image, x + i, y + j, color);
    }
  }
}
RGBABitmapImage *RotateAntiClockwise90Degrees(RGBABitmapImage *image){
  RGBABitmapImage *rotated;
  double x, y;

  rotated = CreateImage(ImageHeight(image), ImageWidth(image), GetBlack());

  for(y = 0.0; y < ImageHeight(image); y = y + 1.0){
    for(x = 0.0; x < ImageWidth(image); x = x + 1.0){
      SetPixel(rotated, y, ImageWidth(image) - 1.0 - x, GetImagePixel(image, x, y));
    }
  }

  return rotated;
}
void DrawCircle(RGBABitmapImage *canvas, double xCenter, double yCenter, double radius, RGBA *color){
  DrawCircleBasicAlgorithm(canvas, xCenter, yCenter, radius, color);
}
void BresenhamsCircleDrawingAlgorithm(RGBABitmapImage *canvas, double xCenter, double yCenter, double radius, RGBA *color){
  double x, y, delta;

  y = radius;
  x = 0.0;

  delta = 3.0 - 2.0*radius;
  for(; y >= x; x = x + 1.0){
    DrawLine1px(canvas, xCenter + x, yCenter + y, xCenter + x, yCenter + y, color);
    DrawLine1px(canvas, xCenter + x, yCenter - y, xCenter + x, yCenter - y, color);
    DrawLine1px(canvas, xCenter - x, yCenter + y, xCenter - x, yCenter + y, color);
    DrawLine1px(canvas, xCenter - x, yCenter - y, xCenter - x, yCenter - y, color);

    DrawLine1px(canvas, xCenter - y, yCenter + x, xCenter - y, yCenter + x, color);
    DrawLine1px(canvas, xCenter - y, yCenter - x, xCenter - y, yCenter - x, color);
    DrawLine1px(canvas, xCenter + y, yCenter + x, xCenter + y, yCenter + x, color);
    DrawLine1px(canvas, xCenter + y, yCenter - x, xCenter + y, yCenter - x, color);

    if(delta < 0.0){
      delta = delta + 4.0*x + 6.0;
    }else{
      delta = delta + 4.0*(x - y) + 10.0;
      y = y - 1.0;
    }
  }
}
void DrawCircleMidpointAlgorithm(RGBABitmapImage *canvas, double xCenter, double yCenter, double radius, RGBA *color){
  double d, x, y;

  d = floor((5.0 - radius*4.0)/4.0);
  x = 0.0;
  y = radius;

  for(; x <= y; x = x + 1.0){
    DrawPixel(canvas, xCenter + x, yCenter + y, color);
    DrawPixel(canvas, xCenter + x, yCenter - y, color);
    DrawPixel(canvas, xCenter - x, yCenter + y, color);
    DrawPixel(canvas, xCenter - x, yCenter - y, color);
    DrawPixel(canvas, xCenter + y, yCenter + x, color);
    DrawPixel(canvas, xCenter + y, yCenter - x, color);
    DrawPixel(canvas, xCenter - y, yCenter + x, color);
    DrawPixel(canvas, xCenter - y, yCenter - x, color);

    if(d < 0.0){
      d = d + 2.0*x + 1.0;
    }else{
      d = d + 2.0*(x - y) + 1.0;
      y = y - 1.0;
    }
  }
}
void DrawCircleBasicAlgorithm(RGBABitmapImage *canvas, double xCenter, double yCenter, double radius, RGBA *color){
  double pixels, a, da, dx, dy;

  /* Place the circle in the center of the pixel. */
  xCenter = floor(xCenter) + 0.5;
  yCenter = floor(yCenter) + 0.5;

  pixels = 2.0*M_PI*radius;

  /* Below a radius of 10 pixels, over-compensate to get a smoother circle. */
  if(radius < 10.0){
    pixels = pixels*10.0;
  }

  da = 2.0*M_PI/pixels;

  for(a = 0.0; a < 2.0*M_PI; a = a + da){
    dx = cos(a)*radius;
    dy = sin(a)*radius;

    /* Floor to get the pixel coordinate. */
    DrawPixel(canvas, floor(xCenter + dx), floor(yCenter + dy), color);
  }
}
void DrawFilledCircle(RGBABitmapImage *canvas, double x, double y, double r, RGBA *color){
  DrawFilledCircleBasicAlgorithm(canvas, x, y, r, color);
}
void DrawFilledCircleMidpointAlgorithm(RGBABitmapImage *canvas, double xCenter, double yCenter, double radius, RGBA *color){
  double d, x, y;

  d = floor((5.0 - radius*4.0)/4.0);
  x = 0.0;
  y = radius;

  for(; x <= y; x = x + 1.0){
    DrawLineBresenhamsAlgorithm(canvas, xCenter + x, yCenter + y, xCenter - x, yCenter + y, color);
    DrawLineBresenhamsAlgorithm(canvas, xCenter + x, yCenter - y, xCenter - x, yCenter - y, color);
    DrawLineBresenhamsAlgorithm(canvas, xCenter + y, yCenter + x, xCenter - y, yCenter + x, color);
    DrawLineBresenhamsAlgorithm(canvas, xCenter + y, yCenter - x, xCenter - y, yCenter - x, color);

    if(d < 0.0){
      d = d + 2.0*x + 1.0;
    }else{
      d = d + 2.0*(x - y) + 1.0;
      y = y - 1.0;
    }
  }
}
void DrawFilledCircleBasicAlgorithm(RGBABitmapImage *canvas, double xCenter, double yCenter, double radius, RGBA *color){
  double pixels, a, da, dx, dy;

  /* Place the circle in the center of the pixel. */
  xCenter = floor(xCenter) + 0.5;
  yCenter = floor(yCenter) + 0.5;

  pixels = 2.0*M_PI*radius;

  /* Below a radius of 10 pixels, over-compensate to get a smoother circle. */
  if(radius < 10.0){
    pixels = pixels*10.0;
  }

  da = 2.0*M_PI/pixels;

  /* Draw lines for a half-circle to fill an entire circle. */
  for(a = 0.0; a < M_PI; a = a + da){
    dx = cos(a)*radius;
    dy = sin(a)*radius;

    /* Floor to get the pixel coordinate. */
    DrawVerticalLine1px(canvas, floor(xCenter - dx), floor(yCenter - dy), floor(2.0*dy) + 1.0, color);
  }
}
void DrawTriangle(RGBABitmapImage *canvas, double xCenter, double yCenter, double height, RGBA *color){
  double x1, y1, x2, y2, x3, y3;

  x1 = floor(xCenter + 0.5);
  y1 = floor(floor(yCenter + 0.5) - height);
  x2 = x1 - 2.0*height*tan(M_PI/6.0);
  y2 = floor(y1 + 2.0*height);
  x3 = x1 + 2.0*height*tan(M_PI/6.0);
  y3 = floor(y1 + 2.0*height);

  DrawLine1px(canvas, x1, y1, x2, y2, color);
  DrawLine1px(canvas, x1, y1, x3, y3, color);
  DrawLine1px(canvas, x2, y2, x3, y3, color);
}
void DrawFilledTriangle(RGBABitmapImage *canvas, double xCenter, double yCenter, double height, RGBA *color){
  double i, offset, x1, y1;

  x1 = floor(xCenter + 0.5);
  y1 = floor(floor(yCenter + 0.5) - height);

  for(i = 0.0; i <= 2.0*height; i = i + 1.0){
    offset = floor(i*tan(M_PI/6.0));
    DrawHorizontalLine1px(canvas, x1 - offset, y1 + i, 2.0*offset, color);
  }
}
void DrawLine(RGBABitmapImage *canvas, double x1, double y1, double x2, double y2, double thickness, RGBA *color){
  DrawLineBresenhamsAlgorithmThick(canvas, x1, y1, x2, y2, thickness, color);
}
void DrawLineBresenhamsAlgorithmThick(RGBABitmapImage *canvas, double x1, double y1, double x2, double y2, double thickness, RGBA *color){
  double x, y, dx, dy, incX, incY, pdx, pdy, es, el, err, t, r;

  dx = x2 - x1;
  dy = y2 - y1;

  incX = Sign(dx);
  incY = Sign(dy);

  dx = abs(dx);
  dy = abs(dy);

  if(dx > dy){
    pdx = incX;
    pdy = 0.0;
    es = dy;
    el = dx;
  }else{
    pdx = 0.0;
    pdy = incY;
    es = dx;
    el = dy;
  }

  x = x1;
  y = y1;
  err = el/2.0;

  if(thickness >= 3.0){
    r = thickness/2.0;
    DrawCircle(canvas, x, y, r, color);
  }else if(floor(thickness) == 2.0){
    DrawFilledRectangle(canvas, x, y, 2.0, 2.0, color);
  }else if(floor(thickness) == 1.0){
    DrawPixel(canvas, x, y, color);
  }

  for(t = 0.0; t < el; t = t + 1.0){
    err = err - es;
    if(err < 0.0){
      err = err + el;
      x = x + incX;
      y = y + incY;
    }else{
      x = x + pdx;
      y = y + pdy;
    }

    if(thickness >= 3.0){
      r = thickness/2.0;
      DrawCircle(canvas, x, y, r, color);
    }else if(floor(thickness) == 2.0){
      DrawFilledRectangle(canvas, x, y, 2.0, 2.0, color);
    }else if(floor(thickness) == 1.0){
      DrawPixel(canvas, x, y, color);
    }
  }
}
void DrawLineBresenhamsAlgorithm(RGBABitmapImage *canvas, double x1, double y1, double x2, double y2, RGBA *color){
  double x, y, dx, dy, incX, incY, pdx, pdy, es, el, err, t;

  dx = x2 - x1;
  dy = y2 - y1;

  incX = Sign(dx);
  incY = Sign(dy);

  dx = abs(dx);
  dy = abs(dy);

  if(dx > dy){
    pdx = incX;
    pdy = 0.0;
    es = dy;
    el = dx;
  }else{
    pdx = 0.0;
    pdy = incY;
    es = dx;
    el = dy;
  }

  x = x1;
  y = y1;
  err = el/2.0;
  DrawPixel(canvas, x, y, color);

  for(t = 0.0; t < el; t = t + 1.0){
    err = err - es;
    if(err < 0.0){
      err = err + el;
      x = x + incX;
      y = y + incY;
    }else{
      x = x + pdx;
      y = y + pdy;
    }

    DrawPixel(canvas, x, y, color);
  }
}
void DrawLineBresenhamsAlgorithmThickPatterned(RGBABitmapImage *canvas, double x1, double y1, double x2, double y2, double thickness, vector<bool> *pattern, NumberReference *offset, RGBA *color){
  double x, y, dx, dy, incX, incY, pdx, pdy, es, el, err, t, r;

  dx = x2 - x1;
  dy = y2 - y1;

  incX = Sign(dx);
  incY = Sign(dy);

  dx = abs(dx);
  dy = abs(dy);

  if(dx > dy){
    pdx = incX;
    pdy = 0.0;
    es = dy;
    el = dx;
  }else{
    pdx = 0.0;
    pdy = incY;
    es = dx;
    el = dy;
  }

  x = x1;
  y = y1;
  err = el/2.0;

  offset->numberValue = fmod(offset->numberValue + 1.0, pattern->size()*thickness);

  if(pattern->at(floor(offset->numberValue/thickness))){
    if(thickness >= 3.0){
      r = thickness/2.0;
      DrawCircle(canvas, x, y, r, color);
    }else if(floor(thickness) == 2.0){
      DrawFilledRectangle(canvas, x, y, 2.0, 2.0, color);
    }else if(floor(thickness) == 1.0){
      DrawPixel(canvas, x, y, color);
    }
  }

  for(t = 0.0; t < el; t = t + 1.0){
    err = err - es;
    if(err < 0.0){
      err = err + el;
      x = x + incX;
      y = y + incY;
    }else{
      x = x + pdx;
      y = y + pdy;
    }

    offset->numberValue = fmod(offset->numberValue + 1.0, pattern->size()*thickness);

    if(pattern->at(floor(offset->numberValue/thickness))){
      if(thickness >= 3.0){
        r = thickness/2.0;
        DrawCircle(canvas, x, y, r, color);
      }else if(floor(thickness) == 2.0){
        DrawFilledRectangle(canvas, x, y, 2.0, 2.0, color);
      }else if(floor(thickness) == 1.0){
        DrawPixel(canvas, x, y, color);
      }
    }
  }
}
vector<bool> *GetLinePattern5(){
  vector<bool> *pattern;

  pattern = new vector<bool> (19.0);

  pattern->at(0) = true;
  pattern->at(1) = true;
  pattern->at(2) = true;
  pattern->at(3) = true;
  pattern->at(4) = true;
  pattern->at(5) = true;
  pattern->at(6) = true;
  pattern->at(7) = true;
  pattern->at(8) = true;
  pattern->at(9) = true;
  pattern->at(10) = false;
  pattern->at(11) = false;
  pattern->at(12) = false;
  pattern->at(13) = true;
  pattern->at(14) = true;
  pattern->at(15) = true;
  pattern->at(16) = false;
  pattern->at(17) = false;
  pattern->at(18) = false;

  return pattern;
}
vector<bool> *GetLinePattern4(){
  vector<bool> *pattern;

  pattern = new vector<bool> (13.0);

  pattern->at(0) = true;
  pattern->at(1) = true;
  pattern->at(2) = true;
  pattern->at(3) = true;
  pattern->at(4) = true;
  pattern->at(5) = true;
  pattern->at(6) = true;
  pattern->at(7) = true;
  pattern->at(8) = true;
  pattern->at(9) = true;
  pattern->at(10) = false;
  pattern->at(11) = false;
  pattern->at(12) = false;

  return pattern;
}
vector<bool> *GetLinePattern3(){
  vector<bool> *pattern;

  pattern = new vector<bool> (13.0);

  pattern->at(0) = true;
  pattern->at(1) = true;
  pattern->at(2) = true;
  pattern->at(3) = true;
  pattern->at(4) = true;
  pattern->at(5) = true;
  pattern->at(6) = false;
  pattern->at(7) = false;
  pattern->at(8) = false;
  pattern->at(9) = true;
  pattern->at(10) = true;
  pattern->at(11) = false;
  pattern->at(12) = false;

  return pattern;
}
vector<bool> *GetLinePattern2(){
  vector<bool> *pattern;

  pattern = new vector<bool> (4.0);

  pattern->at(0) = true;
  pattern->at(1) = true;
  pattern->at(2) = false;
  pattern->at(3) = false;

  return pattern;
}
vector<bool> *GetLinePattern1(){
  vector<bool> *pattern;

  pattern = new vector<bool> (8.0);

  pattern->at(0) = true;
  pattern->at(1) = true;
  pattern->at(2) = true;
  pattern->at(3) = true;
  pattern->at(4) = true;
  pattern->at(5) = false;
  pattern->at(6) = false;
  pattern->at(7) = false;

  return pattern;
}
RGBABitmapImage *Blur(RGBABitmapImage *src, double pixels){
  RGBABitmapImage *dst;
  double x, y, w, h;

  w = ImageWidth(src);
  h = ImageHeight(src);
  dst = CreateImage(w, h, GetTransparent());

  for(x = 0.0; x < w; x = x + 1.0){
    for(y = 0.0; y < h; y = y + 1.0){
      SetPixel(dst, x, y, CreateBlurForPoint(src, x, y, pixels));
    }
  }

  return dst;
}
RGBA *CreateBlurForPoint(RGBABitmapImage *src, double x, double y, double pixels){
  RGBA *rgba;
  double i, j, countColor, countTransparent;
  double fromx, tox, fromy, toy;
  double w, h;
  double alpha;

  w = src->x->size();
  h = src->x->at(0)->y->size();

  rgba = new RGBA();
  rgba->r = 0.0;
  rgba->g = 0.0;
  rgba->b = 0.0;
  rgba->a = 0.0;

  fromx = x - pixels;
  fromx = fmax(fromx, 0.0);

  tox = x + pixels;
  tox = fmin(tox, w - 1.0);

  fromy = y - pixels;
  fromy = fmax(fromy, 0.0);

  toy = y + pixels;
  toy = fmin(toy, h - 1.0);

  countColor = 0.0;
  countTransparent = 0.0;
  for(i = fromx; i < tox; i = i + 1.0){
    for(j = fromy; j < toy; j = j + 1.0){
      alpha = src->x->at(i)->y->at(j)->a;
      if(alpha > 0.0){
        rgba->r = rgba->r + src->x->at(i)->y->at(j)->r;
        rgba->g = rgba->g + src->x->at(i)->y->at(j)->g;
        rgba->b = rgba->b + src->x->at(i)->y->at(j)->b;
        countColor = countColor + 1.0;
      }
      rgba->a = rgba->a + alpha;
      countTransparent = countTransparent + 1.0;
    }
  }

  if(countColor > 0.0){
    rgba->r = rgba->r/countColor;
    rgba->g = rgba->g/countColor;
    rgba->b = rgba->b/countColor;
  }else{
    rgba->r = 0.0;
    rgba->g = 0.0;
    rgba->b = 0.0;
  }

  if(countTransparent > 0.0){
    rgba->a = rgba->a/countTransparent;
  }else{
    rgba->a = 0.0;
  }

  return rgba;
}
vector<wchar_t> *CreateStringScientificNotationDecimalFromNumber(double decimal){
  return CreateStringScientificNotationDecimalFromNumberAllOptions(decimal, false);
}
vector<wchar_t> *CreateStringScientificNotationDecimalFromNumber15d2e(double decimal){
  return CreateStringScientificNotationDecimalFromNumberAllOptions(decimal, true);
}
vector<wchar_t> *CreateStringScientificNotationDecimalFromNumberAllOptions(double decimal, bool complete){
  StringReference *mantissaReference, *exponentReference;
  double multiplier, inc, i, additional;
  double exponent;
  bool done, isPositive, isPositiveExponent;
  vector<wchar_t> *result;

  mantissaReference = new StringReference();
  exponentReference = new StringReference();
  result = new vector<wchar_t> (0.0);
  done = false;
  exponent = 0.0;

  if(decimal < 0.0){
    isPositive = false;
    decimal =  -decimal;
  }else{
    isPositive = true;
  }

  if(decimal == 0.0){
    done = true;
  }

  if( !done ){
    multiplier = 0.0;
    inc = 0.0;

    if(decimal < 1.0){
      multiplier = 10.0;
      inc =  -1.0;
    }else if(decimal >= 10.0){
      multiplier = 0.1;
      inc = 1.0;
    }else{
      done = true;
    }

    if( !done ){
      exponent = round(log10(decimal));
      exponent = fmin(99.0, exponent);
      exponent = fmax( -99.0, exponent);

      decimal = decimal/pow(10.0, exponent);

      /* Adjust */
      for(; (decimal >= 10.0 || decimal < 1.0) && abs(exponent) < 99.0; ){
        decimal = decimal*multiplier;
        exponent = exponent + inc;
      }
    }
  }

  CreateStringFromNumberWithCheck(decimal, 10.0, mantissaReference);

  isPositiveExponent = exponent >= 0.0;
  if( !isPositiveExponent ){
    exponent =  -exponent;
  }

  CreateStringFromNumberWithCheck(exponent, 10.0, exponentReference);

  if( !isPositive ){
    result = AppendString(result, toVector(L"-"));
  }else if(complete){
    result = AppendString(result, toVector(L"+"));
  }

  result = AppendString(result, mantissaReference->string);
  if(complete){
    additional = 16.0;

    if(mantissaReference->string->size() == 1.0){
      result = AppendString(result, toVector(L"."));
      additional = additional - 1.0;
    }

    for(i = mantissaReference->string->size(); i < additional; i = i + 1.0){
      result = AppendString(result, toVector(L"0"));
    }
  }
  result = AppendString(result, toVector(L"e"));

  if( !isPositiveExponent ){
    result = AppendString(result, toVector(L"-"));
  }else if(complete){
    result = AppendString(result, toVector(L"+"));
  }

  if(complete){
    for(i = exponentReference->string->size(); i < 2.0; i = i + 1.0){
      result = AppendString(result, toVector(L"0"));
    }
  }
  result = AppendString(result, exponentReference->string);

  return result;
}
vector<wchar_t> *CreateStringDecimalFromNumber(double decimal){
  StringReference *stringReference;

  stringReference = new StringReference();

  /* This will succeed because base = 10. */
  CreateStringFromNumberWithCheck(decimal, 10.0, stringReference);

  return stringReference->string;
}
bool CreateStringFromNumberWithCheck(double decimal, double base, StringReference *stringReference){
  vector<wchar_t> *string;
  double maximumDigits;
  double digitPosition;
  bool hasPrintedPoint, isPositive;
  double i, d;
  bool success;
  CharacterReference *characterReference;
  wchar_t c;

  isPositive = true;

  if(decimal < 0.0){
    isPositive = false;
    decimal =  -decimal;
  }

  if(decimal == 0.0){
    stringReference->string = toVector(L"0");
    success = true;
  }else{
    characterReference = new CharacterReference();

    if(IsInteger(base)){
      success = true;

      string = new vector<wchar_t> (0.0);

      maximumDigits = GetMaximumDigitsForBase(base);

      digitPosition = GetFirstDigitPosition(decimal, base);

      decimal = round(decimal*pow(base, maximumDigits - digitPosition - 1.0));

      hasPrintedPoint = false;

      if( !isPositive ){
        string = AppendCharacter(string, '-');
      }

      /* Print leading zeros. */
      if(digitPosition < 0.0){
        string = AppendCharacter(string, '0');
        string = AppendCharacter(string, '.');
        hasPrintedPoint = true;
        for(i = 0.0; i <  -digitPosition - 1.0; i = i + 1.0){
          string = AppendCharacter(string, '0');
        }
      }

      /* Print number. */
      for(i = 0.0; i < maximumDigits && success; i = i + 1.0){
        d = floor(decimal/pow(base, maximumDigits - i - 1.0));

        if(d >= base){
          d = base - 1.0;
        }

        if( !hasPrintedPoint  && digitPosition - i + 1.0 == 0.0){
          if(decimal != 0.0){
            string = AppendCharacter(string, '.');
          }
          hasPrintedPoint = true;
        }

        if(decimal == 0.0 && hasPrintedPoint){
        }else{
          success = GetSingleDigitCharacterFromNumberWithCheck(d, base, characterReference);
          if(success){
            c = characterReference->characterValue;
            string = AppendCharacter(string, c);
          }
        }

        if(success){
          decimal = decimal - d*pow(base, maximumDigits - i - 1.0);
        }
      }

      if(success){
        /* Print trailing zeros. */
        for(i = 0.0; i < digitPosition - maximumDigits + 1.0; i = i + 1.0){
          string = AppendCharacter(string, '0');
        }

        stringReference->string = string;
      }
    }else{
      success = false;
    }
  }

  /* Done */
  return success;
}
double GetMaximumDigitsForBase(double base){
  double t;

  t = pow(10.0, 15.0);
  return floor(log10(t)/log10(base));
}
double GetFirstDigitPosition(double decimal, double base){
  double power;
  double t;

  power = ceil(log10(decimal)/log10(base));

  t = decimal*pow(base,  -power);
  if(t < base && t >= 1.0){
  }else if(t >= base){
    power = power + 1.0;
  }else if(t < 1.0){
    power = power - 1.0;
  }

  return power;
}
bool GetSingleDigitCharacterFromNumberWithCheck(double c, double base, CharacterReference *characterReference){
  vector<wchar_t> *numberTable;
  bool success;

  numberTable = GetDigitCharacterTable();

  if(c < base || c < numberTable->size()){
    success = true;
    characterReference->characterValue = numberTable->at(c);
  }else{
    success = false;
  }

  return success;
}
vector<wchar_t> *GetDigitCharacterTable(){
  vector<wchar_t> *numberTable;

  numberTable = toVector(L"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ");

  return numberTable;
}
bool CreateNumberFromDecimalStringWithCheck(vector<wchar_t> *string, NumberReference *decimalReference, StringReference *errorMessage){
  return CreateNumberFromStringWithCheck(string, 10.0, decimalReference, errorMessage);
}
double CreateNumberFromDecimalString(vector<wchar_t> *string){
  NumberReference *doubleReference;
  StringReference *stringReference;
  double number;

  doubleReference = CreateNumberReference(0.0);
  stringReference = CreateStringReference(toVector(L""));
  CreateNumberFromStringWithCheck(string, 10.0, doubleReference, stringReference);
  number = doubleReference->numberValue;

  delete doubleReference;
  delete stringReference;

  return number;
}
bool CreateNumberFromStringWithCheck(vector<wchar_t> *string, double base, NumberReference *numberReference, StringReference *errorMessage){
  bool success;
  BooleanReference *numberIsPositive, *exponentIsPositive;
  NumberArrayReference *beforePoint, *afterPoint, *exponent;

  numberIsPositive = CreateBooleanReference(true);
  exponentIsPositive = CreateBooleanReference(true);
  beforePoint = new NumberArrayReference();
  afterPoint = new NumberArrayReference();
  exponent = new NumberArrayReference();

  if(base >= 2.0 && base <= 36.0){
    success = ExtractPartsFromNumberString(string, base, numberIsPositive, beforePoint, afterPoint, exponentIsPositive, exponent, errorMessage);

    if(success){
      numberReference->numberValue = CreateNumberFromParts(base, numberIsPositive->booleanValue, beforePoint->numberArray, afterPoint->numberArray, exponentIsPositive->booleanValue, exponent->numberArray);
    }
  }else{
    success = false;
    errorMessage->string = toVector(L"Base must be from 2 to 36.");
  }

  return success;
}
double CreateNumberFromParts(double base, bool numberIsPositive, vector<double> *beforePoint, vector<double> *afterPoint, bool exponentIsPositive, vector<double> *exponent){
  double n, i, p, e;

  n = 0.0;

  for(i = 0.0; i < beforePoint->size(); i = i + 1.0){
    p = beforePoint->at(beforePoint->size() - i - 1.0);

    n = n + p*pow(base, i);
  }

  for(i = 0.0; i < afterPoint->size(); i = i + 1.0){
    p = afterPoint->at(i);

    n = n + p*pow(base,  -(i + 1.0));
  }

  if(exponent->size() > 0.0){
    e = 0.0;
    for(i = 0.0; i < exponent->size(); i = i + 1.0){
      p = exponent->at(exponent->size() - i - 1.0);

      e = e + p*pow(base, i);
    }

    if( !exponentIsPositive ){
      e =  -e;
    }

    n = n*pow(base, e);
  }

  if( !numberIsPositive ){
    n =  -n;
  }

  return n;
}
bool ExtractPartsFromNumberString(vector<wchar_t> *n, double base, BooleanReference *numberIsPositive, NumberArrayReference *beforePoint, NumberArrayReference *afterPoint, BooleanReference *exponentIsPositive, NumberArrayReference *exponent, StringReference *errorMessages){
  double i;
  bool success;

  i = 0.0;

  if(i < n->size()){
    if(n->at(i) == '-'){
      numberIsPositive->booleanValue = false;
      i = i + 1.0;
    }else if(n->at(i) == '+'){
      numberIsPositive->booleanValue = true;
      i = i + 1.0;
    }

    success = ExtractPartsFromNumberStringFromSign(n, base, i, beforePoint, afterPoint, exponentIsPositive, exponent, errorMessages);
  }else{
    success = false;
    errorMessages->string = toVector(L"Number cannot have length zero.");
  }

  return success;
}
bool ExtractPartsFromNumberStringFromSign(vector<wchar_t> *n, double base, double i, NumberArrayReference *beforePoint, NumberArrayReference *afterPoint, BooleanReference *exponentIsPositive, NumberArrayReference *exponent, StringReference *errorMessages){
  bool success, done;
  double count, j;

  done = false;
  count = 0.0;
  for(; i + count < n->size() &&  !done ; ){
    if(CharacterIsNumberCharacterInBase(n->at(i + count), base)){
      count = count + 1.0;
    }else{
      done = true;
    }
  }

  if(count >= 1.0){
    beforePoint->numberArray = new vector<double> (count);

    for(j = 0.0; j < count; j = j + 1.0){
      beforePoint->numberArray->at(j) = GetNumberFromNumberCharacterForBase(n->at(i + j), base);
    }

    i = i + count;

    if(i < n->size()){
      success = ExtractPartsFromNumberStringFromPointOrExponent(n, base, i, afterPoint, exponentIsPositive, exponent, errorMessages);
    }else{
      afterPoint->numberArray = new vector<double> (0.0);
      exponent->numberArray = new vector<double> (0.0);
      success = true;
    }
  }else{
    success = false;
    errorMessages->string = toVector(L"Number must have at least one number after the optional sign.");
  }

  return success;
}
bool ExtractPartsFromNumberStringFromPointOrExponent(vector<wchar_t> *n, double base, double i, NumberArrayReference *afterPoint, BooleanReference *exponentIsPositive, NumberArrayReference *exponent, StringReference *errorMessages){
  bool success, done;
  double count, j;

  if(n->at(i) == '.'){
    i = i + 1.0;

    if(i < n->size()){
      done = false;
      count = 0.0;
      for(; i + count < n->size() &&  !done ; ){
        if(CharacterIsNumberCharacterInBase(n->at(i + count), base)){
          count = count + 1.0;
        }else{
          done = true;
        }
      }

      if(count >= 1.0){
        afterPoint->numberArray = new vector<double> (count);

        for(j = 0.0; j < count; j = j + 1.0){
          afterPoint->numberArray->at(j) = GetNumberFromNumberCharacterForBase(n->at(i + j), base);
        }

        i = i + count;

        if(i < n->size()){
          success = ExtractPartsFromNumberStringFromExponent(n, base, i, exponentIsPositive, exponent, errorMessages);
        }else{
          exponent->numberArray = new vector<double> (0.0);
          success = true;
        }
      }else{
        success = false;
        errorMessages->string = toVector(L"There must be at least one digit after the decimal point.");
      }
    }else{
      success = false;
      errorMessages->string = toVector(L"There must be at least one digit after the decimal point.");
    }
  }else if(base <= 14.0 && (n->at(i) == 'e' || n->at(i) == 'E')){
    if(i < n->size()){
      success = ExtractPartsFromNumberStringFromExponent(n, base, i, exponentIsPositive, exponent, errorMessages);
      afterPoint->numberArray = new vector<double> (0.0);
    }else{
      success = false;
      errorMessages->string = toVector(L"There must be at least one digit after the exponent.");
    }
  }else{
    success = false;
    errorMessages->string = toVector(L"Expected decimal point or exponent symbol.");
  }

  return success;
}
bool ExtractPartsFromNumberStringFromExponent(vector<wchar_t> *n, double base, double i, BooleanReference *exponentIsPositive, NumberArrayReference *exponent, StringReference *errorMessages){
  bool success, done;
  double count, j;

  if(base <= 14.0 && (n->at(i) == 'e' || n->at(i) == 'E')){
    i = i + 1.0;

    if(i < n->size()){
      if(n->at(i) == '-'){
        exponentIsPositive->booleanValue = false;
        i = i + 1.0;
      }else if(n->at(i) == '+'){
        exponentIsPositive->booleanValue = true;
        i = i + 1.0;
      }

      if(i < n->size()){
        done = false;
        count = 0.0;
        for(; i + count < n->size() &&  !done ; ){
          if(CharacterIsNumberCharacterInBase(n->at(i + count), base)){
            count = count + 1.0;
          }else{
            done = true;
          }
        }

        if(count >= 1.0){
          exponent->numberArray = new vector<double> (count);

          for(j = 0.0; j < count; j = j + 1.0){
            exponent->numberArray->at(j) = GetNumberFromNumberCharacterForBase(n->at(i + j), base);
          }

          i = i + count;

          if(i == n->size()){
            success = true;
          }else{
            success = false;
            errorMessages->string = toVector(L"There cannot be any characters past the exponent of the number.");
          }
        }else{
          success = false;
          errorMessages->string = toVector(L"There must be at least one digit after the decimal point.");
        }
      }else{
        success = false;
        errorMessages->string = toVector(L"There must be at least one digit after the exponent symbol.");
      }
    }else{
      success = false;
      errorMessages->string = toVector(L"There must be at least one digit after the exponent symbol.");
    }
  }else{
    success = false;
    errorMessages->string = toVector(L"Expected exponent symbol.");
  }

  return success;
}
double GetNumberFromNumberCharacterForBase(wchar_t c, double base){
  vector<wchar_t> *numberTable;
  double i;
  double position;

  numberTable = GetDigitCharacterTable();
  position = 0.0;

  for(i = 0.0; i < base; i = i + 1.0){
    if(numberTable->at(i) == c){
      position = i;
    }
  }

  return position;
}
bool CharacterIsNumberCharacterInBase(wchar_t c, double base){
  vector<wchar_t> *numberTable;
  double i;
  bool found;

  numberTable = GetDigitCharacterTable();
  found = false;

  for(i = 0.0; i < base; i = i + 1.0){
    if(numberTable->at(i) == c){
      found = true;
    }
  }

  return found;
}
vector<double> *StringToNumberArray(vector<wchar_t> *str){
  NumberArrayReference *numberArrayReference;
  StringReference *stringReference;
  vector<double> *numbers;

  numberArrayReference = new NumberArrayReference();
  stringReference = new StringReference();

  StringToNumberArrayWithCheck(str, numberArrayReference, stringReference);

  numbers = numberArrayReference->numberArray;

  delete numberArrayReference;
  delete stringReference;

  return numbers;
}
bool StringToNumberArrayWithCheck(vector<wchar_t> *str, NumberArrayReference *numberArrayReference, StringReference *errorMessage){
  vector<StringReference*> *numberStrings;
  vector<double> *numbers;
  double i;
  vector<wchar_t> *numberString, *trimmedNumberString;
  bool success;
  NumberReference *numberReference;

  numberStrings = SplitByString(str, toVector(L","));

  numbers = new vector<double> (numberStrings->size());
  success = true;
  numberReference = new NumberReference();

  for(i = 0.0; i < numberStrings->size(); i = i + 1.0){
    numberString = numberStrings->at(i)->string;
    trimmedNumberString = Trim(numberString);
    success = CreateNumberFromDecimalStringWithCheck(trimmedNumberString, numberReference, errorMessage);
    numbers->at(i) = numberReference->numberValue;

    FreeStringReference(numberStrings->at(i));
    delete trimmedNumberString;
  }

  delete numberStrings;
  delete numberReference;

  numberArrayReference->numberArray = numbers;

  return success;
}
double Negate(double x){
  return  -x;
}
double Positive(double x){
  return  +x;
}
double Factorial(double x){
  double i, f;

  f = 1.0;

  for(i = 2.0; i <= x; i = i + 1.0){
    f = f*i;
  }

  return f;
}
double Round(double x){
  return floor(x + 0.5);
}
double BankersRound(double x){
  double r;

  if(Absolute(x - Truncate(x)) == 0.5){
    if( !DivisibleBy(Round(x), 2.0) ){
      r = Round(x) - 1.0;
    }else{
      r = Round(x);
    }
  }else{
    r = Round(x);
  }

  return r;
}
double Ceil(double x){
  return ceil(x);
}
double Floor(double x){
  return floor(x);
}
double Truncate(double x){
  double t;

  if(x >= 0.0){
    t = floor(x);
  }else{
    t = ceil(x);
  }

  return t;
}
double Absolute(double x){
  return abs(x);
}
double Logarithm(double x){
  return log10(x);
}
double NaturalLogarithm(double x){
  return log(x);
}
double Sin(double x){
  return sin(x);
}
double Cos(double x){
  return cos(x);
}
double Tan(double x){
  return tan(x);
}
double Asin(double x){
  return asin(x);
}
double Acos(double x){
  return acos(x);
}
double Atan(double x){
  return atan(x);
}
double Atan2(double y, double x){
  double a;

  /* Atan2 is an invalid operation when x = 0 and y = 0, but this method does not return errors. */
  a = 0.0;

  if(x > 0.0){
    a = Atan(y/x);
  }else if(x < 0.0 && y >= 0.0){
    a = Atan(y/x) + M_PI;
  }else if(x < 0.0 && y < 0.0){
    a = Atan(y/x) - M_PI;
  }else if(x == 0.0 && y > 0.0){
    a = M_PI/2.0;
  }else if(x == 0.0 && y < 0.0){
    a =  -M_PI/2.0;
  }

  return a;
}
double Squareroot(double x){
  return sqrt(x);
}
double Exp(double x){
  return exp(x);
}
bool DivisibleBy(double a, double b){
  return ((fmod(a, b)) == 0.0);
}
double Combinations(double n, double k){
  double i, j, c;

  c = 1.0;
  j = 1.0;
  i = n - k + 1.0;

  for(; i <= n; ){
    c = c*i;
    c = c/j;

    i = i + 1.0;
    j = j + 1.0;
  }

  return c;
}
double Permutations(double n, double k){
  double i, c;

  c = 1.0;

  for(i = n - k + 1.0; i <= n; i = i + 1.0){
    c = c*i;
  }

  return c;
}
bool EpsilonCompare(double a, double b, double epsilon){
  return abs(a - b) < epsilon;
}
double GreatestCommonDivisor(double a, double b){
  double t;

  for(; b != 0.0; ){
    t = b;
    b = fmod(a, b);
    a = t;
  }

  return a;
}
double GCDWithSubtraction(double a, double b){
  double g;

  if(a == 0.0){
    g = b;
  }else{
    for(; b != 0.0; ){
      if(a > b){
        a = a - b;
      }else{
        b = b - a;
      }
    }

    g = a;
  }

  return g;
}
bool IsInteger(double a){
  return (a - floor(a)) == 0.0;
}
bool GreatestCommonDivisorWithCheck(double a, double b, NumberReference *gcdReference){
  bool success;
  double gcd;

  if(IsInteger(a) && IsInteger(b)){
    gcd = GreatestCommonDivisor(a, b);
    gcdReference->numberValue = gcd;
    success = true;
  }else{
    success = false;
  }

  return success;
}
double LeastCommonMultiple(double a, double b){
  double lcm;

  if(a > 0.0 && b > 0.0){
    lcm = abs(a*b)/GreatestCommonDivisor(a, b);
  }else{
    lcm = 0.0;
  }

  return lcm;
}
double Sign(double a){
  double s;

  if(a > 0.0){
    s = 1.0;
  }else if(a < 0.0){
    s =  -1.0;
  }else{
    s = 0.0;
  }

  return s;
}
double Max(double a, double b){
  return fmax(a, b);
}
double Min(double a, double b){
  return fmin(a, b);
}
double Power(double a, double b){
  return pow(a, b);
}
double Gamma(double x){
  return LanczosApproximation(x);
}
double LogGamma(double x){
  return log(Gamma(x));
}
double LanczosApproximation(double z){
  vector<double> *p;
  double i, y, t, x;

  p = new vector<double> (8.0);
  p->at(0) = 676.5203681218851;
  p->at(1) =  -1259.1392167224028;
  p->at(2) = 771.32342877765313;
  p->at(3) =  -176.61502916214059;
  p->at(4) = 12.507343278686905;
  p->at(5) =  -0.13857109526572012;
  p->at(6) = 9.9843695780195716e-6;
  p->at(7) = 1.5056327351493116e-7;

  if(z < 0.5){
    y = M_PI/(sin(M_PI*z)*LanczosApproximation(1.0 - z));
  }else{
    z = z - 1.0;
    x = 0.99999999999980993;
    for(i = 0.0; i < p->size(); i = i + 1.0){
      x = x + p->at(i)/(z + i + 1.0);
    }
    t = z + p->size() - 0.5;
    y = sqrt(2.0*M_PI)*pow(t, z + 0.5)*exp( -t)*x;
  }

  return y;
}
double Beta(double x, double y){
  return Gamma(x)*Gamma(y)/Gamma(x + y);
}
double Sinh(double x){
  return (exp(x) - exp( -x))/2.0;
}
double Cosh(double x){
  return (exp(x) + exp( -x))/2.0;
}
double Tanh(double x){
  return Sinh(x)/Cosh(x);
}
double Cot(double x){
  return 1.0/tan(x);
}
double Sec(double x){
  return 1.0/cos(x);
}
double Csc(double x){
  return 1.0/sin(x);
}
double Coth(double x){
  return Cosh(x)/Sinh(x);
}
double Sech(double x){
  return 1.0/Cosh(x);
}
double Csch(double x){
  return 1.0/Sinh(x);
}
double Error(double x){
  double y, t, tau, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10;

  if(x == 0.0){
    y = 0.0;
  }else if(x < 0.0){
    y =  -Error( -x);
  }else{
    c1 =  -1.26551223;
    c2 =  +1.00002368;
    c3 =  +0.37409196;
    c4 =  +0.09678418;
    c5 =  -0.18628806;
    c6 =  +0.27886807;
    c7 =  -1.13520398;
    c8 =  +1.48851587;
    c9 =  -0.82215223;
    c10 =  +0.17087277;

    t = 1.0/(1.0 + 0.5*abs(x));

    tau = t*exp( -pow(x, 2.0) + c1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + t*(c6 + t*(c7 + t*(c8 + t*(c9 + t*c10)))))))));

    y = 1.0 - tau;
  }

  return y;
}
double ErrorInverse(double x){
  double y, a, t;

  a = (8.0*(M_PI - 3.0))/(3.0*M_PI*(4.0 - M_PI));

  t = 2.0/(M_PI*a) + log(1.0 - pow(x, 2.0))/2.0;
  y = Sign(x)*sqrt(sqrt(pow(t, 2.0) - log(1.0 - pow(x, 2.0))/a) - t);

  return y;
}
double FallingFactorial(double x, double n){
  double k, y;

  y = 1.0;

  for(k = 0.0; k <= n - 1.0; k = k + 1.0){
    y = y*(x - k);
  }

  return y;
}
double RisingFactorial(double x, double n){
  double k, y;

  y = 1.0;

  for(k = 0.0; k <= n - 1.0; k = k + 1.0){
    y = y*(x + k);
  }

  return y;
}
double Hypergeometric(double a, double b, double c, double z, double maxIterations, double precision){
  double y;

  if(abs(z) >= 0.5){
    y = pow(1.0 - z,  -a)*HypergeometricDirect(a, c - b, c, z/(z - 1.0), maxIterations, precision);
  }else{
    y = HypergeometricDirect(a, b, c, z, maxIterations, precision);
  }

  return y;
}
double HypergeometricDirect(double a, double b, double c, double z, double maxIterations, double precision){
  double y, yp, n;
  bool done;

  y = 0.0;
  done = false;

  for(n = 0.0; n < maxIterations &&  !done ; n = n + 1.0){
    yp = RisingFactorial(a, n)*RisingFactorial(b, n)/RisingFactorial(c, n)*pow(z, n)/Factorial(n);
    if(abs(yp) < precision){
      done = true;
    }
    y = y + yp;
  }

  return y;
}
double BernouilliNumber(double n){
  return AkiyamaTanigawaAlgorithm(n);
}
double AkiyamaTanigawaAlgorithm(double n){
  double m, j, B;
  vector<double> *A;

  A = new vector<double> (n + 1.0);

  for(m = 0.0; m <= n; m = m + 1.0){
    A->at(m) = 1.0/(m + 1.0);
    for(j = m; j >= 1.0; j = j - 1.0){
      A->at(j - 1.0) = j*(A->at(j - 1.0) - A->at(j));
    }
  }

  B = A->at(0);

  delete A;

  return B;
}
vector<double> *aStringToNumberArray(vector<wchar_t> *string){
  double i;
  vector<double> *array;

  array = new vector<double> (string->size());

  for(i = 0.0; i < string->size(); i = i + 1.0){
    array->at(i) = string->at(i);
  }
  return array;
}
vector<wchar_t> *aNumberArrayToString(vector<double> *array){
  double i;
  vector<wchar_t> *string;

  string = new vector<wchar_t> (array->size());

  for(i = 0.0; i < array->size(); i = i + 1.0){
    string->at(i) = array->at(i);
  }
  return string;
}
bool aNumberArraysEqual(vector<double> *a, vector<double> *b){
  bool equal;
  double i;

  equal = true;
  if(a->size() == b->size()){
    for(i = 0.0; i < a->size() && equal; i = i + 1.0){
      if(a->at(i) != b->at(i)){
        equal = false;
      }
    }
  }else{
    equal = false;
  }

  return equal;
}
bool aBooleanArraysEqual(vector<bool> *a, vector<bool> *b){
  bool equal;
  double i;

  equal = true;
  if(a->size() == b->size()){
    for(i = 0.0; i < a->size() && equal; i = i + 1.0){
      if(a->at(i) != b->at(i)){
        equal = false;
      }
    }
  }else{
    equal = false;
  }

  return equal;
}
bool aStringsEqual(vector<wchar_t> *a, vector<wchar_t> *b){
  bool equal;
  double i;

  equal = true;
  if(a->size() == b->size()){
    for(i = 0.0; i < a->size() && equal; i = i + 1.0){
      if(a->at(i) != b->at(i)){
        equal = false;
      }
    }
  }else{
    equal = false;
  }

  return equal;
}
void aFillNumberArray(vector<double> *a, double value){
  double i;

  for(i = 0.0; i < a->size(); i = i + 1.0){
    a->at(i) = value;
  }
}
void aFillString(vector<wchar_t> *a, wchar_t value){
  double i;

  for(i = 0.0; i < a->size(); i = i + 1.0){
    a->at(i) = value;
  }
}
void aFillBooleanArray(vector<bool> *a, bool value){
  double i;

  for(i = 0.0; i < a->size(); i = i + 1.0){
    a->at(i) = value;
  }
}
bool aFillNumberArrayRange(vector<double> *a, double value, double from, double to){
  double i, length;
  bool success;

  if(from >= 0.0 && from <= a->size() && to >= 0.0 && to <= a->size() && from <= to){
    length = to - from;
    for(i = 0.0; i < length; i = i + 1.0){
      a->at(from + i) = value;
    }

    success = true;
  }else{
    success = false;
  }

  return success;
}
bool aFillBooleanArrayRange(vector<bool> *a, bool value, double from, double to){
  double i, length;
  bool success;

  if(from >= 0.0 && from <= a->size() && to >= 0.0 && to <= a->size() && from <= to){
    length = to - from;
    for(i = 0.0; i < length; i = i + 1.0){
      a->at(from + i) = value;
    }

    success = true;
  }else{
    success = false;
  }

  return success;
}
bool aFillStringRange(vector<wchar_t> *a, wchar_t value, double from, double to){
  double i, length;
  bool success;

  if(from >= 0.0 && from <= a->size() && to >= 0.0 && to <= a->size() && from <= to){
    length = to - from;
    for(i = 0.0; i < length; i = i + 1.0){
      a->at(from + i) = value;
    }

    success = true;
  }else{
    success = false;
  }

  return success;
}
vector<double> *aCopyNumberArray(vector<double> *a){
  double i;
  vector<double> *n;

  n = new vector<double> (a->size());

  for(i = 0.0; i < a->size(); i = i + 1.0){
    n->at(i) = a->at(i);
  }

  return n;
}
vector<bool> *aCopyBooleanArray(vector<bool> *a){
  double i;
  vector<bool> *n;

  n = new vector<bool> (a->size());

  for(i = 0.0; i < a->size(); i = i + 1.0){
    n->at(i) = a->at(i);
  }

  return n;
}
vector<wchar_t> *aCopyString(vector<wchar_t> *a){
  double i;
  vector<wchar_t> *n;

  n = new vector<wchar_t> (a->size());

  for(i = 0.0; i < a->size(); i = i + 1.0){
    n->at(i) = a->at(i);
  }

  return n;
}
bool aCopyNumberArrayRange(vector<double> *a, double from, double to, NumberArrayReference *copyReference){
  double i, length;
  vector<double> *n;
  bool success;

  if(from >= 0.0 && from <= a->size() && to >= 0.0 && to <= a->size() && from <= to){
    length = to - from;
    n = new vector<double> (length);

    for(i = 0.0; i < length; i = i + 1.0){
      n->at(i) = a->at(from + i);
    }

    copyReference->numberArray = n;
    success = true;
  }else{
    success = false;
  }

  return success;
}
bool aCopyBooleanArrayRange(vector<bool> *a, double from, double to, BooleanArrayReference *copyReference){
  double i, length;
  vector<bool> *n;
  bool success;

  if(from >= 0.0 && from <= a->size() && to >= 0.0 && to <= a->size() && from <= to){
    length = to - from;
    n = new vector<bool> (length);

    for(i = 0.0; i < length; i = i + 1.0){
      n->at(i) = a->at(from + i);
    }

    copyReference->booleanArray = n;
    success = true;
  }else{
    success = false;
  }

  return success;
}
bool aCopyStringRange(vector<wchar_t> *a, double from, double to, StringReference *copyReference){
  double i, length;
  vector<wchar_t> *n;
  bool success;

  if(from >= 0.0 && from <= a->size() && to >= 0.0 && to <= a->size() && from <= to){
    length = to - from;
    n = new vector<wchar_t> (length);

    for(i = 0.0; i < length; i = i + 1.0){
      n->at(i) = a->at(from + i);
    }

    copyReference->string = n;
    success = true;
  }else{
    success = false;
  }

  return success;
}
bool aIsLastElement(double length, double index){
  return index + 1.0 == length;
}
vector<double> *aCreateNumberArray(double length, double value){
  vector<double> *array;

  array = new vector<double> (length);
  aFillNumberArray(array, value);

  return array;
}
vector<bool> *aCreateBooleanArray(double length, bool value){
  vector<bool> *array;

  array = new vector<bool> (length);
  aFillBooleanArray(array, value);

  return array;
}
vector<wchar_t> *aCreateString(double length, wchar_t value){
  vector<wchar_t> *array;

  array = new vector<wchar_t> (length);
  aFillString(array, value);

  return array;
}
void aSwapElementsOfNumberArray(vector<double> *A, double ai, double bi){
  double tmp;

  tmp = A->at(ai);
  A->at(ai) = A->at(bi);
  A->at(bi) = tmp;
}
void aSwapElementsOfStringArray(StringArrayReference *A, double ai, double bi){
  StringReference *tmp;

  tmp = A->stringArray->at(ai);
  A->stringArray->at(ai) = A->stringArray->at(bi);
  A->stringArray->at(bi) = tmp;
}
void aReverseNumberArray(vector<double> *array){
  double i;

  for(i = 0.0; i < array->size()/2.0; i = i + 1.0){
    aSwapElementsOfNumberArray(array, i, array->size() - i - 1.0);
  }
}
BooleanReference *CreateBooleanReference(bool value){
  BooleanReference *ref;

  ref = new BooleanReference();
  ref->booleanValue = value;

  return ref;
}
BooleanArrayReference *CreateBooleanArrayReference(vector<bool> *value){
  BooleanArrayReference *ref;

  ref = new BooleanArrayReference();
  ref->booleanArray = value;

  return ref;
}
BooleanArrayReference *CreateBooleanArrayReferenceLengthValue(double length, bool value){
  BooleanArrayReference *ref;
  double i;

  ref = new BooleanArrayReference();
  ref->booleanArray = new vector<bool> (length);

  for(i = 0.0; i < length; i = i + 1.0){
    ref->booleanArray->at(i) = value;
  }

  return ref;
}
void FreeBooleanArrayReference(BooleanArrayReference *booleanArrayReference){
  delete booleanArrayReference->booleanArray;
  delete booleanArrayReference;
}
CharacterReference *CreateCharacterReference(wchar_t value){
  CharacterReference *ref;

  ref = new CharacterReference();
  ref->characterValue = value;

  return ref;
}
NumberReference *CreateNumberReference(double value){
  NumberReference *ref;

  ref = new NumberReference();
  ref->numberValue = value;

  return ref;
}
NumberArrayReference *CreateNumberArrayReference(vector<double> *value){
  NumberArrayReference *ref;

  ref = new NumberArrayReference();
  ref->numberArray = value;

  return ref;
}
NumberArrayReference *CreateNumberArrayReferenceLengthValue(double length, double value){
  NumberArrayReference *ref;
  double i;

  ref = new NumberArrayReference();
  ref->numberArray = new vector<double> (length);

  for(i = 0.0; i < length; i = i + 1.0){
    ref->numberArray->at(i) = value;
  }

  return ref;
}
void FreeNumberArrayReference(NumberArrayReference *numberArrayReference){
  delete numberArrayReference->numberArray;
  delete numberArrayReference;
}
StringReference *CreateStringReference(vector<wchar_t> *value){
  StringReference *ref;

  ref = new StringReference();
  ref->string = value;

  return ref;
}
StringReference *CreateStringReferenceLengthValue(double length, wchar_t value){
  StringReference *ref;
  double i;

  ref = new StringReference();
  ref->string = new vector<wchar_t> (length);

  for(i = 0.0; i < length; i = i + 1.0){
    ref->string->at(i) = value;
  }

  return ref;
}
void FreeStringReference(StringReference *stringReference){
  delete stringReference->string;
  delete stringReference;
}
StringArrayReference *CreateStringArrayReference(vector<StringReference*> *strings){
  StringArrayReference *ref;

  ref = new StringArrayReference();
  ref->stringArray = strings;

  return ref;
}
StringArrayReference *CreateStringArrayReferenceLengthValue(double length, vector<wchar_t> *value){
  StringArrayReference *ref;
  double i;

  ref = new StringArrayReference();
  ref->stringArray = new vector<StringReference*> (length);

  for(i = 0.0; i < length; i = i + 1.0){
    ref->stringArray->at(i) = CreateStringReference(value);
  }

  return ref;
}
void FreeStringArrayReference(StringArrayReference *stringArrayReference){
  double i;

  for(i = 0.0; i < stringArrayReference->stringArray->size(); i = i + 1.0){
    delete stringArrayReference->stringArray->at(i);
  }
  delete stringArrayReference->stringArray;
  delete stringArrayReference;
}
vector<wchar_t> *GetPixelFontData(){
  return toVector(L"00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000011000000110000000000000000000000110000001100000011000000110000001100000011000000110000000000000000000000000000000000000000000000000000000000000000000000000000110110001101100011011000110110000000000000000000000000001100110011001101111111101100110011001101111111101100110011001100000000000000000000000000000000000011000011111101111111111011000111110000111111000011111000110111111111101111110000110000000000000000000011100001101100011011011011101100000110000011000001100000110111011011011000110110000111000000000000000001111111001100011111100110001101100001110000011100001101100110011001100110011011000011100000000000000000000000000000000000000000000000000000000000000000000000000000110000011100000110000011100000000000000000000001100000001100000001100000011000000110000001100000011000000110000001100000110000011000000000000000000000000110000011000001100000011000000110000001100000011000000110000001100000001100000001100000000000000000000000000000000001001100101011010001111001111111100111100010110101001100100000000000000000000000000000000000000000001100000011000000110001111111111111111000110000001100000011000000000000000000000000000000000000000110000011000001110000011100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000111111111111111100000000000000000000000000000000000000000000000000000000000000000001110000011100000000000000000000000000000000000000000000000000000000000000000000000000000001100000011000001100000011000001100000011000001100000011000001100000011000001100000011000000000000000000000000111100011001101100001111000111110011111101101111110011111000111100001101100110001111000000000000000000011111100001100000011000000110000001100000011000000110000001100000011110000111000001100000000000000000001111111100000011000000110000011000001100000110000011000001100000110000001110011101111110000000000000000001111110111001111100000011000000111000000111111011100000110000001100000011100111011111100000000000000000001100000011000000110000001100000011000011111111001100110011011000111100001110000011000000000000000000000111111011100111110000001100000011100000011111110000001100000011000000110000001111111111000000000000000001111110111001111100001111000011111000110111111100000011000000110000001111100111011111100000000000000000000011000000110000001100000011000001100000110000011000001100000011000000110000001111111100000000000000000111111011100111110000111100001111100111011111101110011111000011110000111110011101111110000000000000000001111110111001111100000011000000110000001111111011100111110000111100001111100111011111100000000000000000000000000001110000011100000000000000000000011100000111000000000000000000000000000000000000000000000000000000110000011000001110000011100000000000000000000011100000111000000000000000000000000000000000000000000001100000001100000001100000001100000001100000001100000110000011000001100000110000011000000000000000000000000000000000000011111111111111110000000011111111111111110000000000000000000000000000000000000000000000000000011000001100000110000011000001100000110000000110000000110000000110000000110000000110000000000000000000011000000000000000000000011000000110000011000001100000110000001100001111000011011111100000000000000000111111000000011011110011110110111100101110111011110000110111111000000000000000000000000000000000000000001100001111000011110000111100001111111111110000111100001111000011011001100011110000011000000000000000000001111111111000111100001111000011111000110111111111100011110000111100001111100011011111110000000000000000011111101110011100000011000000110000001100000011000000110000001100000011111001110111111000000000000000000011111101110011111000111100001111000011110000111100001111000011111000110111001100111111000000000000000011111111000000110000001100000011000000110011111100000011000000110000001100000011111111110000000000000000000000110000001100000011000000110000001100000011001111110000001100000011000000111111111100000000000000000111111011100111110000111100001111110011000000110000001100000011000000111110011101111110000000000000000011000011110000111100001111000011110000111111111111000011110000111100001111000011110000110000000000000000011111100001100000011000000110000001100000011000000110000001100000011000000110000111111000000000000000000011111001110111011000110110000001100000011000000110000001100000011000000110000001100000000000000000000011000011011000110011001100011011000011110000011100001111000110110011001101100011110000110000000000000000111111110000001100000011000000110000001100000011000000110000001100000011000000110000001100000000000000001100001111000011110000111100001111000011110000111101101111111111111111111110011111000011000000000000000011100011111000111111001111110011111110111101101111011111110011111100111111000111110001110000000000000000011111101110011111000011110000111100001111000011110000111100001111000011111001110111111000000000000000000000001100000011000000110000001100000011011111111110001111000011110000111110001101111111000000000000000011111100011101101111101111011011110000111100001111000011110000111100001101100110001111000000000000000000110000110110001100110011000110110000111101111111111000111100001111000011111000110111111100000000000000000111111011100111110000001100000011100000011111100000011100000011000000111110011101111110000000000000000000011000000110000001100000011000000110000001100000011000000110000001100000011000111111110000000000000000011111101110011111000011110000111100001111000011110000111100001111000011110000111100001100000000000000000001100000111100001111000110011001100110110000111100001111000011110000111100001111000011000000000000000011000011111001111111111111111111110110111101101111000011110000111100001111000011110000110000000000000000110000110110011001100110001111000011110000011000001111000011110001100110011001101100001100000000000000000001100000011000000110000001100000011000000110000011110000111100011001100110011011000011000000000000000011111111000000110000001100000110000011000111111000110000011000001100000011000000111111110000000000000000001111000000110000001100000011000000110000001100000011000000110000001100000011000011110000000000110000001100000001100000011000000011000000110000000110000001100000001100000011000000011000000110000000000000000000111100001100000011000000110000001100000011000000110000001100000011000000110000001111000000000000000000000000000000000000000000000000000000000000000000000000001100001101100110001111000001100011111111111111110000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000110000001110000001100000011100000000000000000111111101100001111000011111111101100000011000011011111100000000000000000000000000000000000000000000000000111111111000011110000111100001111000011011111110000001100000011000000110000001100000011000000000000000001111110110000110000001100000011000000111100001101111110000000000000000000000000000000000000000000000000111111101100001111000011110000111100001111111110110000001100000011000000110000001100000000000000000000001111111000000011000000110111111111000011110000110111111000000000000000000000000000000000000000000000000000001100000011000000110000001100000011000011111100001100000011000000110011001100011110000111111011000011110000001100000011111110110000111100001111000011011111100000000000000000000000000000000000000000000000001100001111000011110000111100001111000011110000110111111100000011000000110000001100000011000000000000000000011000000110000001100000011000000110000001100000011000000000000000000000011000000000000001110000110110001100000011000000110000001100000011000000110000001100000000000000000000001100000000000000000000000000000110001100110011000111110000111100011011001100110110001100000011000000110000001100000011000000000000000001111110000110000001100000011000000110000001100000011000000110000001100000011000000111100000000000000000110110111101101111011011110110111101101111011011011111110000000000000000000000000000000000000000000000000110001101100011011000110110001101100011011000110011111100000000000000000000000000000000000000000000000000111110011000110110001101100011011000110110001100111110000000000000000000000000000000000000001100000011000000110111111111000011110000111100001111000011011111110000000000000000000000000000000011000000110000001100000011111110110000111100001111000011110000111111111000000000000000000000000000000000000000000000000000000011000000110000001100000011000000110000011101111111000000000000000000000000000000000000000000000000011111111100000011000000011111100000001100000011111111100000000000000000000000000000000000000000000000000011100001101100000011000000110000001100000011000011111100001100000011000000110000000000000000000000000001111110011000110110001101100011011000110110001101100011000000000000000000000000000000000000000000000000000110000011110000111100011001100110011011000011110000110000000000000000000000000000000000000000000000001100001111100111111111111101101111000011110000111100001100000000000000000000000000000000000000000000000011000011011001100011110000011000001111000110011011000011000000000000000000000000000000000000001100000110000001100000110000011000001111000110011001100110110000110000000000000000000000000000000000000000000000001111111100000110000011000001100000110000011000001111111100000000000000000000000000000000000000000000000011110000000110000001100000011000000111000000111100011100000110000001100000011000111100000001100000011000000110000001100000011000000110000001100000011000000110000001100000011000000110000001100000000000000000000000111100011000000110000001100000111000111100000011100000011000000110000001100000001111");
}
void DrawAsciiCharacter(RGBABitmapImage *image, double topx, double topy, wchar_t a, RGBA *color){
  double index, x, y, pixel, basis, ybasis;
  vector<wchar_t> *allCharData;

  index = a;
  index = index - 32.0;
  allCharData = GetPixelFontData();

  basis = index*8.0*13.0;

  for(y = 0.0; y < 13.0; y = y + 1.0){
    ybasis = basis + y*8.0;
    for(x = 0.0; x < 8.0; x = x + 1.0){
      pixel = allCharData->at(ybasis + x);
      if(pixel == '1'){
        DrawPixel(image, topx + 8.0 - 1.0 - x, topy + 13.0 - 1.0 - y, color);
      }
    }
  }
}
double GetTextWidth(vector<wchar_t> *text){
  double charWidth, spacing, width;

  charWidth = 8.0;
  spacing = 2.0;

  if(text->size() == 0.0){
    width = 0.0;
  }else{
    width = text->size()*charWidth + (text->size() - 1.0)*spacing;
  }

  return width;
}
double GetTextHeight(vector<wchar_t> *text){
  return 13.0;
}
void AssertFalse(bool b, NumberReference *failures){
  if(b){
    failures->numberValue = failures->numberValue + 1.0;
  }
}
void AssertTrue(bool b, NumberReference *failures){
  if( !b ){
    failures->numberValue = failures->numberValue + 1.0;
  }
}
void AssertEquals(double a, double b, NumberReference *failures){
  if(a != b){
    failures->numberValue = failures->numberValue + 1.0;
  }
}
void AssertBooleansEqual(bool a, bool b, NumberReference *failures){
  if(a != b){
    failures->numberValue = failures->numberValue + 1.0;
  }
}
void AssertCharactersEqual(wchar_t a, wchar_t b, NumberReference *failures){
  if(a != b){
    failures->numberValue = failures->numberValue + 1.0;
  }
}
void AssertStringEquals(vector<wchar_t> *a, vector<wchar_t> *b, NumberReference *failures){
  if( !aStringsEqual(a, b) ){
    failures->numberValue = failures->numberValue + 1.0;
  }
}
void AssertNumberArraysEqual(vector<double> *a, vector<double> *b, NumberReference *failures){
  double i;

  if(a->size() == b->size()){
    for(i = 0.0; i < a->size(); i = i + 1.0){
      AssertEquals(a->at(i), b->at(i), failures);
    }
  }else{
    failures->numberValue = failures->numberValue + 1.0;
  }
}
void AssertBooleanArraysEqual(vector<bool> *a, vector<bool> *b, NumberReference *failures){
  double i;

  if(a->size() == b->size()){
    for(i = 0.0; i < a->size(); i = i + 1.0){
      AssertBooleansEqual(a->at(i), b->at(i), failures);
    }
  }else{
    failures->numberValue = failures->numberValue + 1.0;
  }
}
void AssertStringArraysEqual(vector<StringReference*> *a, vector<StringReference*> *b, NumberReference *failures){
  double i;

  if(a->size() == b->size()){
    for(i = 0.0; i < a->size(); i = i + 1.0){
      AssertStringEquals(a->at(i)->string, b->at(i)->string, failures);
    }
  }else{
    failures->numberValue = failures->numberValue + 1.0;
  }
}
vector<double> *ConvertToPNG(RGBABitmapImage *image){
  return ConvertToPNGWithOptions(image, 6.0, false, 0.0, 0.001);
}
vector<double> *ConvertToPNGGrayscale(RGBABitmapImage *image){
  return ConvertToPNGWithOptions(image, 0.0, false, 0.0, 0.001);
}
PHYS *PysicsHeader(double pixelsPerMeter){
  PHYS *phys;

  phys = new PHYS();

  phys->pixelsPerMeter = pixelsPerMeter;

  return phys;
}
vector<double> *ConvertToPNGWithOptions(RGBABitmapImage *image, double colorType, bool setPhys, double pixelsPerMeter, double compressionLevel){
  PNGImage *png;
  vector<double> *pngData, *colorData;

  png = new PNGImage();

  png->signature = PNGSignature();

  png->ihdr = PNGHeader(image, colorType);

  png->physPresent = setPhys;
  png->phys = PysicsHeader(pixelsPerMeter);

  if(colorType == 6.0){
    colorData = GetPNGColorData(image);
  }else{
    colorData = GetPNGColorDataGreyscale(image);
  }
  png->zlibStruct = ZLibCompressStaticHuffman(colorData, compressionLevel);

  pngData = PNGSerializeChunks(png);

  return pngData;
}
vector<double> *PNGSerializeChunks(PNGImage *png){
  double length, i, chunkLength;
  vector<double> *data;
  NumberReference *position;

  length = png->signature->size() + 12.0 + PNGHeaderLength() + 12.0 + PNGIDATLength(png) + 12.0;
  if(png->physPresent){
    length = length + 4.0 + 4.0 + 1.0 + 12.0;
  }
  data = new vector<double> (length);
  position = CreateNumberReference(0.0);

  /* Signature */
  for(i = 0.0; i < png->signature->size(); i = i + 1.0){
    WriteByte(data, png->signature->at(i), position);
  }

  /* Header */
  chunkLength = PNGHeaderLength();
  Write4BytesBE(data, chunkLength, position);
  WriteStringBytes(data, toVector(L"IHDR"), position);
  Write4BytesBE(data, png->ihdr->Width, position);
  Write4BytesBE(data, png->ihdr->Height, position);
  WriteByte(data, png->ihdr->BitDepth, position);
  WriteByte(data, png->ihdr->ColourType, position);
  WriteByte(data, png->ihdr->CompressionMethod, position);
  WriteByte(data, png->ihdr->FilterMethod, position);
  WriteByte(data, png->ihdr->InterlaceMethod, position);
  Write4BytesBE(data, CRC32OfInterval(data, position->numberValue - chunkLength - 4.0, chunkLength + 4.0), position);

  /* pHYs */
  if(png->physPresent){
    chunkLength = 4.0 + 4.0 + 1.0;
    Write4BytesBE(data, chunkLength, position);
    WriteStringBytes(data, toVector(L"pHYs"), position);

    Write4BytesBE(data, png->phys->pixelsPerMeter, position);
    Write4BytesBE(data, png->phys->pixelsPerMeter, position);
    WriteByte(data, 1.0, position);
    /* 1 = pixels per meter */
    Write4BytesBE(data, CRC32OfInterval(data, position->numberValue - chunkLength - 4.0, chunkLength + 4.0), position);
  }

  /* IDAT */
  chunkLength = PNGIDATLength(png);
  Write4BytesBE(data, chunkLength, position);
  WriteStringBytes(data, toVector(L"IDAT"), position);
  WriteByte(data, png->zlibStruct->CMF, position);
  WriteByte(data, png->zlibStruct->FLG, position);
  for(i = 0.0; i < png->zlibStruct->CompressedDataBlocks->size(); i = i + 1.0){
    WriteByte(data, png->zlibStruct->CompressedDataBlocks->at(i), position);
  }
  Write4BytesBE(data, png->zlibStruct->Adler32CheckValue, position);
  Write4BytesBE(data, CRC32OfInterval(data, position->numberValue - chunkLength - 4.0, chunkLength + 4.0), position);

  /* IEND */
  chunkLength = 0.0;
  Write4BytesBE(data, chunkLength, position);
  WriteStringBytes(data, toVector(L"IEND"), position);
  Write4BytesBE(data, CRC32OfInterval(data, position->numberValue - 4.0, 4.0), position);

  return data;
}
double PNGIDATLength(PNGImage *png){
  return 2.0 + png->zlibStruct->CompressedDataBlocks->size() + 4.0;
}
double PNGHeaderLength(){
  return 4.0 + 4.0 + 1.0 + 1.0 + 1.0 + 1.0 + 1.0;
}
vector<double> *GetPNGColorData(RGBABitmapImage *image){
  vector<double> *colordata;
  double length, x, y, next;
  RGBA *rgba;

  length = 4.0*ImageWidth(image)*ImageHeight(image) + ImageHeight(image);

  colordata = new vector<double> (length);

  next = 0.0;

  for(y = 0.0; y < ImageHeight(image); y = y + 1.0){
    colordata->at(next) = 0.0;
    next = next + 1.0;
    for(x = 0.0; x < ImageWidth(image); x = x + 1.0){
      rgba = image->x->at(x)->y->at(y);
      colordata->at(next) = Round(rgba->r*255.0);
      next = next + 1.0;
      colordata->at(next) = Round(rgba->g*255.0);
      next = next + 1.0;
      colordata->at(next) = Round(rgba->b*255.0);
      next = next + 1.0;
      colordata->at(next) = Round(rgba->a*255.0);
      next = next + 1.0;
    }
  }

  return colordata;
}
vector<double> *GetPNGColorDataGreyscale(RGBABitmapImage *image){
  vector<double> *colordata;
  double length, x, y, next;
  RGBA *rgba;

  length = ImageWidth(image)*ImageHeight(image) + ImageHeight(image);

  colordata = new vector<double> (length);

  next = 0.0;

  for(y = 0.0; y < ImageHeight(image); y = y + 1.0){
    colordata->at(next) = 0.0;
    next = next + 1.0;
    for(x = 0.0; x < ImageWidth(image); x = x + 1.0){
      rgba = image->x->at(x)->y->at(y);
      colordata->at(next) = Round(rgba->r*255.0);
      next = next + 1.0;
    }
  }

  return colordata;
}
IHDR *PNGHeader(RGBABitmapImage *image, double colortype){
  IHDR *ihdr;

  ihdr = new IHDR();
  ihdr->Width = ImageWidth(image);
  ihdr->Height = ImageHeight(image);
  /* Truecolour with alpha */
  ihdr->BitDepth = 8.0;
  ihdr->ColourType = colortype;
  ihdr->FilterMethod = 0.0;
  /* None */
  ihdr->CompressionMethod = 0.0;
  /* zlib */
  ihdr->InterlaceMethod = 0.0;
  /* no interlace */
  return ihdr;
}
vector<double> *PNGSignature(){
  vector<double> *s;

  s = new vector<double> (8.0);
  s->at(0) = 137.0;
  s->at(1) = 80.0;
  s->at(2) = 78.0;
  s->at(3) = 71.0;
  s->at(4) = 13.0;
  s->at(5) = 10.0;
  s->at(6) = 26.0;
  s->at(7) = 10.0;

  return s;
}
vector<double> *PNGReadDataChunks(vector<Chunk*> *cs){
  double i, j, length, zlibpos;
  Chunk *c;
  vector<double> *zlibData;

  length = 0.0;
  for(i = 0.0; i < cs->size(); i = i + 1.0){
    c = cs->at(i);
    if(aStringsEqual(c->type, toVector(L"IDAT"))){
      length = length + c->length;
    }
  }

  zlibData = new vector<double> (length);
  zlibpos = 0.0;

  for(i = 0.0; i < cs->size(); i = i + 1.0){
    c = cs->at(i);
    if(aStringsEqual(c->type, toVector(L"IDAT"))){
      for(j = 0.0; j < c->length; j = j + 1.0){
        zlibData->at(zlibpos) = c->data->at(j);
        zlibpos = zlibpos + 1.0;
      }
    }
  }

  return zlibData;
}
bool PNGReadHeader(RGBABitmapImage *image, vector<Chunk*> *cs, StringReference *errorMessages){
  double i;
  IHDR *ihdr;
  Chunk *c;
  NumberReference *position;
  RGBABitmapImage *n;
  bool success;

  position = CreateNumberReference(0.0);
  success = false;

  for(i = 0.0; i < cs->size(); i = i + 1.0){
    c = cs->at(i);
    if(aStringsEqual(c->type, toVector(L"IHDR"))){
      ihdr = new IHDR();

      ihdr->Width = Read4bytesBE(c->data, position);
      ihdr->Height = Read4bytesBE(c->data, position);
      ihdr->BitDepth = ReadByte(c->data, position);
      ihdr->ColourType = ReadByte(c->data, position);
      ihdr->CompressionMethod = ReadByte(c->data, position);
      ihdr->FilterMethod = ReadByte(c->data, position);
      ihdr->InterlaceMethod = ReadByte(c->data, position);

      n = CreateImage(ihdr->Width, ihdr->Height, GetTransparent());
      image->x = n->x;

      if(ihdr->ColourType == 6.0){
        if(ihdr->BitDepth == 8.0){
          if(ihdr->CompressionMethod == 0.0){
            if(ihdr->FilterMethod == 0.0){
              if(ihdr->InterlaceMethod == 0.0){
                success = true;
              }else{
                success = false;
                errorMessages->string = AppendString(errorMessages->string, toVector(L"Interlace method not supported."));
              }
            }else{
              success = false;
              errorMessages->string = AppendString(errorMessages->string, toVector(L"Filter method not supported."));
            }
          }else{
            success = false;
            errorMessages->string = AppendString(errorMessages->string, toVector(L"Compression type not supported."));
          }
        }else{
          success = false;
          errorMessages->string = AppendString(errorMessages->string, toVector(L"Bit depth not supported."));
        }
      }else{
        success = false;
        errorMessages->string = AppendString(errorMessages->string, toVector(L"Color type not supported."));
      }
    }
  }

  return success;
}
vector<Chunk*> *PNGReadChunks(vector<double> *data, NumberReference *position){
  bool done;
  double prepos;
  double chunks;
  Chunk *c;
  vector<Chunk*> *cs;
  double i;
  done = false;
  prepos = position->numberValue;
  for(chunks = 0.0;  !done ; chunks = chunks + 1.0){
    c = PNGReadChunk(data, position);
    if(aStringsEqual(c->type, toVector(L"IEND"))){
      done = true;
    }
  }
  position->numberValue = prepos;
  cs = new vector<Chunk*> (chunks);
  for(i = 0.0; i < chunks; i = i + 1.0){
    cs->at(i) = PNGReadChunk(data, position);
  }

  return cs;
}
Chunk *PNGReadChunk(vector<double> *data, NumberReference *position){
  Chunk *c;

  c = new Chunk();

  c->length = Read4bytesBE(data, position);
  c->type = new vector<wchar_t> (4.0);
  c->type->at(0) = ReadByte(data, position);
  c->type->at(1) = ReadByte(data, position);
  c->type->at(2) = ReadByte(data, position);
  c->type->at(3) = ReadByte(data, position);
  c->data = ReadXbytes(data, position, c->length);
  c->crc = Read4bytesBE(data, position);

  return c;
}
void WriteStringToStingStream(vector<wchar_t> *stream, NumberReference *index, vector<wchar_t> *src){
  double i;

  for(i = 0.0; i < src->size(); i = i + 1.0){
    stream->at(index->numberValue + i) = src->at(i);
  }
  index->numberValue = index->numberValue + src->size();
}
void WriteCharacterToStingStream(vector<wchar_t> *stream, NumberReference *index, wchar_t src){
  stream->at(index->numberValue) = src;
  index->numberValue = index->numberValue + 1.0;
}
void WriteBooleanToStingStream(vector<wchar_t> *stream, NumberReference *index, bool src){
  if(src){
    WriteStringToStingStream(stream, index, toVector(L"true"));
  }else{
    WriteStringToStingStream(stream, index, toVector(L"false"));
  }
}
bool SubstringWithCheck(vector<wchar_t> *string, double from, double to, StringReference *stringReference){
  bool success;

  if(from >= 0.0 && from <= string->size() && to >= 0.0 && to <= string->size() && from <= to){
    stringReference->string = Substring(string, from, to);
    success = true;
  }else{
    success = false;
  }

  return success;
}
vector<wchar_t> *Substring(vector<wchar_t> *string, double from, double to){
  vector<wchar_t> *n;
  double i, length;

  length = to - from;

  n = new vector<wchar_t> (length);

  for(i = from; i < to; i = i + 1.0){
    n->at(i - from) = string->at(i);
  }

  return n;
}
vector<wchar_t> *AppendString(vector<wchar_t> *s1, vector<wchar_t> *s2){
  vector<wchar_t> *newString;

  newString = ConcatenateString(s1, s2);

  delete s1;

  return newString;
}
vector<wchar_t> *ConcatenateString(vector<wchar_t> *s1, vector<wchar_t> *s2){
  vector<wchar_t> *newString;
  double i;

  newString = new vector<wchar_t> (s1->size() + s2->size());

  for(i = 0.0; i < s1->size(); i = i + 1.0){
    newString->at(i) = s1->at(i);
  }

  for(i = 0.0; i < s2->size(); i = i + 1.0){
    newString->at(s1->size() + i) = s2->at(i);
  }

  return newString;
}
vector<wchar_t> *AppendCharacter(vector<wchar_t> *string, wchar_t c){
  vector<wchar_t> *newString;

  newString = ConcatenateCharacter(string, c);

  delete string;

  return newString;
}
vector<wchar_t> *ConcatenateCharacter(vector<wchar_t> *string, wchar_t c){
  vector<wchar_t> *newString;
  double i;
  newString = new vector<wchar_t> (string->size() + 1.0);

  for(i = 0.0; i < string->size(); i = i + 1.0){
    newString->at(i) = string->at(i);
  }

  newString->at(string->size()) = c;

  return newString;
}
vector<StringReference*> *SplitByCharacter(vector<wchar_t> *toSplit, wchar_t splitBy){
  vector<StringReference*> *split;
  vector<wchar_t> *stringToSplitBy;

  stringToSplitBy = new vector<wchar_t> (1.0);
  stringToSplitBy->at(0) = splitBy;

  split = SplitByString(toSplit, stringToSplitBy);

  delete stringToSplitBy;

  return split;
}
bool IndexOfCharacter(vector<wchar_t> *string, wchar_t character, NumberReference *indexReference){
  double i;
  bool found;

  found = false;
  for(i = 0.0; i < string->size() &&  !found ; i = i + 1.0){
    if(string->at(i) == character){
      found = true;
      indexReference->numberValue = i;
    }
  }

  return found;
}
bool SubstringEqualsWithCheck(vector<wchar_t> *string, double from, vector<wchar_t> *substring, BooleanReference *equalsReference){
  bool success;

  if(from < string->size()){
    success = true;
    equalsReference->booleanValue = SubstringEquals(string, from, substring);
  }else{
    success = false;
  }

  return success;
}
bool SubstringEquals(vector<wchar_t> *string, double from, vector<wchar_t> *substring){
  double i;
  bool equal;

  equal = true;
  if(string->size() - from >= substring->size()){
    for(i = 0.0; i < substring->size() && equal; i = i + 1.0){
      if(string->at(from + i) != substring->at(i)){
        equal = false;
      }
    }
  }else{
    equal = false;
  }

  return equal;
}
bool IndexOfString(vector<wchar_t> *string, vector<wchar_t> *substring, NumberReference *indexReference){
  double i;
  bool found;

  found = false;
  for(i = 0.0; i < string->size() - substring->size() + 1.0 &&  !found ; i = i + 1.0){
    if(SubstringEquals(string, i, substring)){
      found = true;
      indexReference->numberValue = i;
    }
  }

  return found;
}
bool ContainsCharacter(vector<wchar_t> *string, wchar_t character){
  double i;
  bool found;

  found = false;
  for(i = 0.0; i < string->size() &&  !found ; i = i + 1.0){
    if(string->at(i) == character){
      found = true;
    }
  }

  return found;
}
bool ContainsString(vector<wchar_t> *string, vector<wchar_t> *substring){
  return IndexOfString(string, substring, new NumberReference());
}
void ToUpperCase(vector<wchar_t> *string){
  double i;

  for(i = 0.0; i < string->size(); i = i + 1.0){
    string->at(i) = charToUpperCase(string->at(i));
  }
}
void ToLowerCase(vector<wchar_t> *string){
  double i;

  for(i = 0.0; i < string->size(); i = i + 1.0){
    string->at(i) = charToLowerCase(string->at(i));
  }
}
bool EqualsIgnoreCase(vector<wchar_t> *a, vector<wchar_t> *b){
  bool equal;
  double i;

  if(a->size() == b->size()){
    equal = true;
    for(i = 0.0; i < a->size() && equal; i = i + 1.0){
      if(charToLowerCase(a->at(i)) != charToLowerCase(b->at(i))){
        equal = false;
      }
    }
  }else{
    equal = false;
  }

  return equal;
}
vector<wchar_t> *ReplaceString(vector<wchar_t> *string, vector<wchar_t> *toReplace, vector<wchar_t> *replaceWith){
  vector<wchar_t> *result;
  double i;
  BooleanReference *equalsReference;
  bool success;

  equalsReference = new BooleanReference();
  result = new vector<wchar_t> (0.0);

  for(i = 0.0; i < string->size(); ){
    success = SubstringEqualsWithCheck(string, i, toReplace, equalsReference);
    if(success){
      success = equalsReference->booleanValue;
    }

    if(success && toReplace->size() > 0.0){
      result = ConcatenateString(result, replaceWith);
      i = i + toReplace->size();
    }else{
      result = ConcatenateCharacter(result, string->at(i));
      i = i + 1.0;
    }
  }

  return result;
}
vector<wchar_t> *ReplaceCharacter(vector<wchar_t> *string, wchar_t toReplace, wchar_t replaceWith){
  vector<wchar_t> *result;
  double i;

  result = new vector<wchar_t> (0.0);

  for(i = 0.0; i < string->size(); i = i + 1.0){
    if(string->at(i) == toReplace){
      result = ConcatenateCharacter(result, replaceWith);
    }else{
      result = ConcatenateCharacter(result, string->at(i));
    }
  }

  return result;
}
vector<wchar_t> *Trim(vector<wchar_t> *string){
  vector<wchar_t> *result;
  double i, lastWhitespaceLocationStart, lastWhitespaceLocationEnd;
  bool firstNonWhitespaceFound;

  /* Find whitepaces at the start. */
  lastWhitespaceLocationStart =  -1.0;
  firstNonWhitespaceFound = false;
  for(i = 0.0; i < string->size() &&  !firstNonWhitespaceFound ; i = i + 1.0){
    if(charIsWhiteSpace(string->at(i))){
      lastWhitespaceLocationStart = i;
    }else{
      firstNonWhitespaceFound = true;
    }
  }

  /* Find whitepaces at the end. */
  lastWhitespaceLocationEnd = string->size();
  firstNonWhitespaceFound = false;
  for(i = string->size() - 1.0; i >= 0.0 &&  !firstNonWhitespaceFound ; i = i - 1.0){
    if(charIsWhiteSpace(string->at(i))){
      lastWhitespaceLocationEnd = i;
    }else{
      firstNonWhitespaceFound = true;
    }
  }

  if(lastWhitespaceLocationStart < lastWhitespaceLocationEnd){
    result = Substring(string, lastWhitespaceLocationStart + 1.0, lastWhitespaceLocationEnd);
  }else{
    result = new vector<wchar_t> (0.0);
  }

  return result;
}
bool StartsWith(vector<wchar_t> *string, vector<wchar_t> *start){
  bool startsWithString;

  startsWithString = false;
  if(string->size() >= start->size()){
    startsWithString = SubstringEquals(string, 0.0, start);
  }

  return startsWithString;
}
bool EndsWith(vector<wchar_t> *string, vector<wchar_t> *end){
  bool endsWithString;

  endsWithString = false;
  if(string->size() >= end->size()){
    endsWithString = SubstringEquals(string, string->size() - end->size(), end);
  }

  return endsWithString;
}
vector<StringReference*> *SplitByString(vector<wchar_t> *toSplit, vector<wchar_t> *splitBy){
  vector<StringReference*> *split;
  vector<wchar_t> *next;
  double i;
  wchar_t c;
  StringReference *n;

  split = new vector<StringReference*> (0.0);

  next = new vector<wchar_t> (0.0);
  for(i = 0.0; i < toSplit->size(); ){
    c = toSplit->at(i);

    if(SubstringEquals(toSplit, i, splitBy)){
      n = new StringReference();
      n->string = next;
      split = AddString(split, n);
      next = new vector<wchar_t> (0.0);
      i = i + splitBy->size();
    }else{
      next = AppendCharacter(next, c);
      i = i + 1.0;
    }
  }

  n = new StringReference();
  n->string = next;
  split = AddString(split, n);

  return split;
}
bool StringIsBefore(vector<wchar_t> *a, vector<wchar_t> *b){
  bool before, equal, done;
  double i;

  before = false;
  equal = true;
  done = false;

  if(a->size() == 0.0 && b->size() > 0.0){
    before = true;
  }else{
    for(i = 0.0; i < a->size() && i < b->size() &&  !done ; i = i + 1.0){
      if(a->at(i) != b->at(i)){
        equal = false;
      }
      if(charCharacterIsBefore(a->at(i), b->at(i))){
        before = true;
      }
      if(charCharacterIsBefore(b->at(i), a->at(i))){
        done = true;
      }
    }

    if(equal){
      if(a->size() < b->size()){
        before = true;
      }
    }
  }

  return before;
}
vector<double> *ReadXbytes(vector<double> *data, NumberReference *position, double length){
  vector<double> *r;
  double i;

  r = new vector<double> (length);

  for(i = 0.0; i < length; i = i + 1.0){
    r->at(i) = ReadByte(data, position);
  }

  return r;
}
double Read4bytesBE(vector<double> *data, NumberReference *position){
  double r;

  r = 0.0;
  r = r + pow(2.0, 24.0)*ReadByte(data, position);
  r = r + pow(2.0, 16.0)*ReadByte(data, position);
  r = r + pow(2.0, 8.0)*ReadByte(data, position);
  r = r + ReadByte(data, position);

  return r;
}
double Read2bytesBE(vector<double> *data, NumberReference *position){
  double r;

  r = 0.0;
  r = r + pow(2.0, 8.0)*ReadByte(data, position);
  r = r + ReadByte(data, position);

  return r;
}
double ReadByte(vector<double> *data, NumberReference *position){
  double next;

  next = data->at(position->numberValue);
  position->numberValue = position->numberValue + 1.0;

  return next;
}
double Read4bytesLE(vector<double> *data, NumberReference *position){
  double r;

  r = 0.0;
  r = r + ReadByte(data, position);
  r = r + pow(2.0, 8.0)*ReadByte(data, position);
  r = r + pow(2.0, 16.0)*ReadByte(data, position);
  r = r + pow(2.0, 24.0)*ReadByte(data, position);

  return r;
}
void WriteByte(vector<double> *data, double b, NumberReference *position){
  data->at(position->numberValue) = b;
  position->numberValue = position->numberValue + 1.0;
}
void Write2BytesLE(vector<double> *data, double b, NumberReference *position){
  data->at(position->numberValue) = Round(fmod(b, pow(2.0, 8.0)));
  position->numberValue = position->numberValue + 1.0;
  data->at(position->numberValue) = fmod(floor(b/pow(2.0, 8.0)), pow(2.0, 8.0));
  position->numberValue = position->numberValue + 1.0;
}
void Write4BytesLE(vector<double> *data, double b, NumberReference *position){
  data->at(position->numberValue) = Round(fmod(b, pow(2.0, 8.0)));
  position->numberValue = position->numberValue + 1.0;
  data->at(position->numberValue) = fmod(floor(b/pow(2.0, 8.0)), pow(2.0, 8.0));
  position->numberValue = position->numberValue + 1.0;
  data->at(position->numberValue) = fmod(floor(b/pow(2.0, 16.0)), pow(2.0, 8.0));
  position->numberValue = position->numberValue + 1.0;
  data->at(position->numberValue) = fmod(floor(b/pow(2.0, 24.0)), pow(2.0, 8.0));
  position->numberValue = position->numberValue + 1.0;
}
void Write2BytesBE(vector<double> *data, double b, NumberReference *position){
  data->at(position->numberValue) = fmod(floor(b/pow(2.0, 8.0)), pow(2.0, 8.0));
  position->numberValue = position->numberValue + 1.0;
  data->at(position->numberValue) = Round(fmod(b, pow(2.0, 8.0)));
  position->numberValue = position->numberValue + 1.0;
}
void Write4BytesBE(vector<double> *data, double b, NumberReference *position){
  data->at(position->numberValue) = fmod(floor(b/pow(2.0, 24.0)), pow(2.0, 8.0));
  position->numberValue = position->numberValue + 1.0;
  data->at(position->numberValue) = fmod(floor(b/pow(2.0, 16.0)), pow(2.0, 8.0));
  position->numberValue = position->numberValue + 1.0;
  data->at(position->numberValue) = fmod(floor(b/pow(2.0, 8.0)), pow(2.0, 8.0));
  position->numberValue = position->numberValue + 1.0;
  data->at(position->numberValue) = Round(fmod(b, pow(2.0, 8.0)));
  position->numberValue = position->numberValue + 1.0;
}
void WriteStringBytes(vector<double> *data, vector<wchar_t> *cs, NumberReference *position){
  double i, v;

  for(i = 0.0; i < cs->size(); i = i + 1.0){
    v = cs->at(i);
    WriteByte(data, v, position);
  }
}
vector<double> *MakeCRC32Table(){
  double c, n, k;
  vector<double> *crcTable;

  crcTable = new vector<double> (256.0);

  for(n = 0.0; n < 256.0; n = n + 1.0){
    c = n;
    for(k = 0.0; k < 8.0; k = k + 1.0){
      if( !DivisibleBy(c, 2.0) ){
        c = Xor4Byte(3988292384.0, floor(c/2.0));
      }else{
        c = floor(c/2.0);
      }
    }
    crcTable->at(n) = c;
  }

  return crcTable;
}
double UpdateCRC32(double crc, vector<double> *buf, vector<double> *crc_table){
  double n, index;

  for(n = 0.0; n < buf->size(); n = n + 1.0){
    index = And4Byte(Xor4Byte(crc, buf->at(n)), pow(2.0, 8.0) - 1.0);
    crc = Xor4Byte(crc_table->at(index), floor(crc/pow(2.0, 8.0)));
  }

  return crc;
}
double CalculateCRC32(vector<double> *buf){
  vector<double> *crcTable;
  double b32max, value;

  crcTable = MakeCRC32Table();

  b32max = pow(2.0, 32.0) - 1.0;
  value = UpdateCRC32(b32max, buf, crcTable);

  return Xor4Byte(value, b32max);
}
double CRC32OfInterval(vector<double> *data, double from, double length){
  vector<double> *crcBase;
  double i, crc;

  crcBase = new vector<double> (length);

  for(i = 0.0; i < length; i = i + 1.0){
    crcBase->at(i) = data->at(from + i);
  }

  crc = CalculateCRC32(crcBase);

  delete crcBase;

  return crc;
}
ZLIBStruct *ZLibCompressNoCompression(vector<double> *data){
  ZLIBStruct *zlibStruct;

  zlibStruct = new ZLIBStruct();

  zlibStruct->CMF = 120.0;
  zlibStruct->FLG = 1.0;
  zlibStruct->CompressedDataBlocks = DeflateDataNoCompression(data);
  zlibStruct->Adler32CheckValue = ComputeAdler32(data);

  return zlibStruct;
}
ZLIBStruct *ZLibCompressStaticHuffman(vector<double> *data, double level){
  ZLIBStruct *zlibStruct;

  zlibStruct = new ZLIBStruct();

  zlibStruct->CMF = 120.0;
  zlibStruct->FLG = 1.0;
  zlibStruct->CompressedDataBlocks = DeflateDataStaticHuffman(data, level);
  zlibStruct->Adler32CheckValue = ComputeAdler32(data);

  return zlibStruct;
}
vector<double> *AddNumber(vector<double> *list, double a){
  vector<double> *newlist;
  double i;

  newlist = new vector<double> (list->size() + 1.0);
  for(i = 0.0; i < list->size(); i = i + 1.0){
    newlist->at(i) = list->at(i);
  }
  newlist->at(list->size()) = a;
		
  delete list;
		
  return newlist;
}
void AddNumberRef(NumberArrayReference *list, double i){
  list->numberArray = AddNumber(list->numberArray, i);
}
vector<double> *RemoveNumber(vector<double> *list, double n){
  vector<double> *newlist;
  double i;

  newlist = new vector<double> (list->size() - 1.0);

  if(n >= 0.0 && n < list->size()){
    for(i = 0.0; i < list->size(); i = i + 1.0){
      if(i < n){
        newlist->at(i) = list->at(i);
      }
      if(i > n){
        newlist->at(i - 1.0) = list->at(i);
      }
    }

    delete list;
  }else{
    delete newlist;
  }
		
  return newlist;
}
double GetNumberRef(NumberArrayReference *list, double i){
  return list->numberArray->at(i);
}
void RemoveNumberRef(NumberArrayReference *list, double i){
  list->numberArray = RemoveNumber(list->numberArray, i);
}
vector<StringReference*> *AddString(vector<StringReference*> *list, StringReference *a){
  vector<StringReference*> *newlist;
  double i;

  newlist = new vector<StringReference*> (list->size() + 1.0);

  for(i = 0.0; i < list->size(); i = i + 1.0){
    newlist->at(i) = list->at(i);
  }
  newlist->at(list->size()) = a;
		
  delete list;
		
  return newlist;
}
void AddStringRef(StringArrayReference *list, StringReference *i){
  list->stringArray = AddString(list->stringArray, i);
}
vector<StringReference*> *RemoveString(vector<StringReference*> *list, double n){
  vector<StringReference*> *newlist;
  double i;

  newlist = new vector<StringReference*> (list->size() - 1.0);

  if(n >= 0.0 && n < list->size()){
    for(i = 0.0; i < list->size(); i = i + 1.0){
      if(i < n){
        newlist->at(i) = list->at(i);
      }
      if(i > n){
        newlist->at(i - 1.0) = list->at(i);
      }
    }

    delete list;
  }else{
    delete newlist;
  }
		
  return newlist;
}
StringReference *GetStringRef(StringArrayReference *list, double i){
  return list->stringArray->at(i);
}
void RemoveStringRef(StringArrayReference *list, double i){
  list->stringArray = RemoveString(list->stringArray, i);
}
vector<bool> *AddBoolean(vector<bool> *list, bool a){
  vector<bool> *newlist;
  double i;

  newlist = new vector<bool> (list->size() + 1.0);
  for(i = 0.0; i < list->size(); i = i + 1.0){
    newlist->at(i) = list->at(i);
  }
  newlist->at(list->size()) = a;
		
  delete list;
		
  return newlist;
}
void AddBooleanRef(BooleanArrayReference *list, bool i){
  list->booleanArray = AddBoolean(list->booleanArray, i);
}
vector<bool> *RemoveBoolean(vector<bool> *list, double n){
  vector<bool> *newlist;
  double i;

  newlist = new vector<bool> (list->size() - 1.0);

  if(n >= 0.0 && n < list->size()){
    for(i = 0.0; i < list->size(); i = i + 1.0){
      if(i < n){
        newlist->at(i) = list->at(i);
      }
      if(i > n){
        newlist->at(i - 1.0) = list->at(i);
      }
    }

    delete list;
  }else{
    delete newlist;
  }
		
  return newlist;
}
bool GetBooleanRef(BooleanArrayReference *list, double i){
  return list->booleanArray->at(i);
}
void RemoveDecimalRef(BooleanArrayReference *list, double i){
  list->booleanArray = RemoveBoolean(list->booleanArray, i);
}
LinkedListStrings *CreateLinkedListString(){
  LinkedListStrings *ll;

  ll = new LinkedListStrings();
  ll->first = new LinkedListNodeStrings();
  ll->last = ll->first;
  ll->last->end = true;

  return ll;
}
void LinkedListAddString(LinkedListStrings *ll, vector<wchar_t> *value){
  ll->last->end = false;
  ll->last->value = value;
  ll->last->next = new LinkedListNodeStrings();
  ll->last->next->end = true;
  ll->last = ll->last->next;
}
vector<StringReference*> *LinkedListStringsToArray(LinkedListStrings *ll){
  vector<StringReference*> *array;
  double length, i;
  LinkedListNodeStrings *node;

  node = ll->first;

  length = LinkedListStringsLength(ll);

  array = new vector<StringReference*> (length);

  for(i = 0.0; i < length; i = i + 1.0){
    array->at(i) = new StringReference();
    array->at(i)->string = node->value;
    node = node->next;
  }

  return array;
}
double LinkedListStringsLength(LinkedListStrings *ll){
  double l;
  LinkedListNodeStrings *node;

  l = 0.0;
  node = ll->first;
  for(;  !node->end ; ){
    node = node->next;
    l = l + 1.0;
  }

  return l;
}
void FreeLinkedListString(LinkedListStrings *ll){
  LinkedListNodeStrings *node, *prev;

  node = ll->first;

  for(;  !node->end ; ){
    prev = node;
    node = node->next;
    delete prev;
  }

  delete node;
}
LinkedListNumbers *CreateLinkedListNumbers(){
  LinkedListNumbers *ll;

  ll = new LinkedListNumbers();
  ll->first = new LinkedListNodeNumbers();
  ll->last = ll->first;
  ll->last->end = true;

  return ll;
}
vector<LinkedListNumbers*> *CreateLinkedListNumbersArray(double length){
  vector<LinkedListNumbers*> *lls;
  double i;

  lls = new vector<LinkedListNumbers*> (length);
  for(i = 0.0; i < lls->size(); i = i + 1.0){
    lls->at(i) = CreateLinkedListNumbers();
  }

  return lls;
}
void LinkedListAddNumber(LinkedListNumbers *ll, double value){
  ll->last->end = false;
  ll->last->value = value;
  ll->last->next = new LinkedListNodeNumbers();
  ll->last->next->end = true;
  ll->last = ll->last->next;
}
double LinkedListNumbersLength(LinkedListNumbers *ll){
  double l;
  LinkedListNodeNumbers *node;

  l = 0.0;
  node = ll->first;
  for(;  !node->end ; ){
    node = node->next;
    l = l + 1.0;
  }

  return l;
}
double LinkedListNumbersIndex(LinkedListNumbers *ll, double index){
  double i;
  LinkedListNodeNumbers *node;

  node = ll->first;
  for(i = 0.0; i < index; i = i + 1.0){
    node = node->next;
  }

  return node->value;
}
void LinkedListInsertNumber(LinkedListNumbers *ll, double index, double value){
  double i;
  LinkedListNodeNumbers *node, *tmp;

  if(index == 0.0){
    tmp = ll->first;
    ll->first = new LinkedListNodeNumbers();
    ll->first->next = tmp;
    ll->first->value = value;
    ll->first->end = false;
  }else{
    node = ll->first;
    for(i = 0.0; i < index - 1.0; i = i + 1.0){
      node = node->next;
    }

    tmp = node->next;
    node->next = new LinkedListNodeNumbers();
    node->next->next = tmp;
    node->next->value = value;
    node->next->end = false;
  }
}
void LinkedListSet(LinkedListNumbers *ll, double index, double value){
  double i;
  LinkedListNodeNumbers *node;

  node = ll->first;
  for(i = 0.0; i < index; i = i + 1.0){
    node = node->next;
  }

  node->next->value = value;
}
void LinkedListRemoveNumber(LinkedListNumbers *ll, double index){
  double i;
  LinkedListNodeNumbers *node, *prev;

  node = ll->first;
  prev = ll->first;

  for(i = 0.0; i < index; i = i + 1.0){
    prev = node;
    node = node->next;
  }

  if(index == 0.0){
    ll->first = prev->next;
  }
  if( !prev->next->end ){
    prev->next = prev->next->next;
  }
}
void FreeLinkedListNumbers(LinkedListNumbers *ll){
  LinkedListNodeNumbers *node, *prev;

  node = ll->first;

  for(;  !node->end ; ){
    prev = node;
    node = node->next;
    delete prev;
  }

  delete node;
}
void FreeLinkedListNumbersArray(vector<LinkedListNumbers*> *lls){
  double i;

  for(i = 0.0; i < lls->size(); i = i + 1.0){
    FreeLinkedListNumbers(lls->at(i));
  }
  delete lls;
}
vector<double> *LinkedListNumbersToArray(LinkedListNumbers *ll){
  vector<double> *array;
  double length, i;
  LinkedListNodeNumbers *node;

  node = ll->first;

  length = LinkedListNumbersLength(ll);

  array = new vector<double> (length);

  for(i = 0.0; i < length; i = i + 1.0){
    array->at(i) = node->value;
    node = node->next;
  }

  return array;
}
LinkedListNumbers *ArrayToLinkedListNumbers(vector<double> *array){
  LinkedListNumbers *ll;
  double i;

  ll = CreateLinkedListNumbers();

  for(i = 0.0; i < array->size(); i = i + 1.0){
    LinkedListAddNumber(ll, array->at(i));
  }

  return ll;
}
bool LinkedListNumbersEqual(LinkedListNumbers *a, LinkedListNumbers *b){
  bool equal, done;
  LinkedListNodeNumbers *an, *bn;

  an = a->first;
  bn = b->first;

  equal = true;
  done = false;
  for(; equal &&  !done ; ){
    if(an->end == bn->end){
      if(an->end){
        done = true;
      }else if(an->value == bn->value){
        an = an->next;
        bn = bn->next;
      }else{
        equal = false;
      }
    }else{
      equal = false;
    }
  }

  return equal;
}
LinkedListCharacters *CreateLinkedListCharacter(){
  LinkedListCharacters *ll;

  ll = new LinkedListCharacters();
  ll->first = new LinkedListNodeCharacters();
  ll->last = ll->first;
  ll->last->end = true;

  return ll;
}
void LinkedListAddCharacter(LinkedListCharacters *ll, wchar_t value){
  ll->last->end = false;
  ll->last->value = value;
  ll->last->next = new LinkedListNodeCharacters();
  ll->last->next->end = true;
  ll->last = ll->last->next;
}
vector<wchar_t> *LinkedListCharactersToArray(LinkedListCharacters *ll){
  vector<wchar_t> *array;
  double length, i;
  LinkedListNodeCharacters *node;

  node = ll->first;

  length = LinkedListCharactersLength(ll);

  array = new vector<wchar_t> (length);

  for(i = 0.0; i < length; i = i + 1.0){
    array->at(i) = node->value;
    node = node->next;
  }

  return array;
}
double LinkedListCharactersLength(LinkedListCharacters *ll){
  double l;
  LinkedListNodeCharacters *node;

  l = 0.0;
  node = ll->first;
  for(;  !node->end ; ){
    node = node->next;
    l = l + 1.0;
  }

  return l;
}
void FreeLinkedListCharacter(LinkedListCharacters *ll){
  LinkedListNodeCharacters *node, *prev;

  node = ll->first;

  for(;  !node->end ; ){
    prev = node;
    node = node->next;
    delete prev;
  }

  delete node;
}
DynamicArrayNumbers *CreateDynamicArrayNumbers(){
  DynamicArrayNumbers *da;

  da = new DynamicArrayNumbers();
  da->array = new vector<double> (10.0);
  da->length = 0.0;

  return da;
}
DynamicArrayNumbers *CreateDynamicArrayNumbersWithInitialCapacity(double capacity){
  DynamicArrayNumbers *da;

  da = new DynamicArrayNumbers();
  da->array = new vector<double> (capacity);
  da->length = 0.0;

  return da;
}
void DynamicArrayAddNumber(DynamicArrayNumbers *da, double value){
  if(da->length == da->array->size()){
    DynamicArrayNumbersIncreaseSize(da);
  }

  da->array->at(da->length) = value;
  da->length = da->length + 1.0;
}
void DynamicArrayNumbersIncreaseSize(DynamicArrayNumbers *da){
  double newLength, i;
  vector<double> *newArray;

  newLength = round(da->array->size()*3.0/2.0);
  newArray = new vector<double> (newLength);

  for(i = 0.0; i < da->array->size(); i = i + 1.0){
    newArray->at(i) = da->array->at(i);
  }

  delete da->array;

  da->array = newArray;
}
bool DynamicArrayNumbersDecreaseSizeNecessary(DynamicArrayNumbers *da){
  bool needsDecrease;

  needsDecrease = false;

  if(da->length > 10.0){
    needsDecrease = da->length <= round(da->array->size()*2.0/3.0);
  }

  return needsDecrease;
}
void DynamicArrayNumbersDecreaseSize(DynamicArrayNumbers *da){
  double newLength, i;
  vector<double> *newArray;

  newLength = round(da->array->size()*2.0/3.0);
  newArray = new vector<double> (newLength);

  for(i = 0.0; i < newLength; i = i + 1.0){
    newArray->at(i) = da->array->at(i);
  }

  delete da->array;

  da->array = newArray;
}
double DynamicArrayNumbersIndex(DynamicArrayNumbers *da, double index){
  return da->array->at(index);
}
double DynamicArrayNumbersLength(DynamicArrayNumbers *da){
  return da->length;
}
void DynamicArrayInsertNumber(DynamicArrayNumbers *da, double index, double value){
  double i;

  if(da->length == da->array->size()){
    DynamicArrayNumbersIncreaseSize(da);
  }

  for(i = da->length; i > index; i = i - 1.0){
    da->array->at(i) = da->array->at(i - 1.0);
  }

  da->array->at(index) = value;

  da->length = da->length + 1.0;
}
void DynamicArraySet(DynamicArrayNumbers *da, double index, double value){
  da->array->at(index) = value;
}
void DynamicArrayRemoveNumber(DynamicArrayNumbers *da, double index){
  double i;

  for(i = index; i < da->length - 1.0; i = i + 1.0){
    da->array->at(i) = da->array->at(i + 1.0);
  }

  da->length = da->length - 1.0;

  if(DynamicArrayNumbersDecreaseSizeNecessary(da)){
    DynamicArrayNumbersDecreaseSize(da);
  }
}
void FreeDynamicArrayNumbers(DynamicArrayNumbers *da){
  delete da->array;
  delete da;
}
vector<double> *DynamicArrayNumbersToArray(DynamicArrayNumbers *da){
  vector<double> *array;
  double i;

  array = new vector<double> (da->length);

  for(i = 0.0; i < da->length; i = i + 1.0){
    array->at(i) = da->array->at(i);
  }

  return array;
}
DynamicArrayNumbers *ArrayToDynamicArrayNumbersWithOptimalSize(vector<double> *array){
  DynamicArrayNumbers *da;
  double i;
  double c, n, newCapacity;

  /*
         c = 10*(3/2)^n
         log(c) = log(10*(3/2)^n)
         log(c) = log(10) + log((3/2)^n)
         log(c) = 1 + log((3/2)^n)
         log(c) - 1 = log((3/2)^n)
         log(c) - 1 = n*log(3/2)
         n = (log(c) - 1)/log(3/2)
         */
  c = array->size();
  n = (log(c) - 1.0)/log(3.0/2.0);
  newCapacity = floor(n) + 1.0;

  da = CreateDynamicArrayNumbersWithInitialCapacity(newCapacity);

  for(i = 0.0; i < array->size(); i = i + 1.0){
    da->array->at(i) = array->at(i);
  }

  return da;
}
DynamicArrayNumbers *ArrayToDynamicArrayNumbers(vector<double> *array){
  DynamicArrayNumbers *da;

  da = new DynamicArrayNumbers();
  da->array = aCopyNumberArray(array);
  da->length = array->size();

  return da;
}
bool DynamicArrayNumbersEqual(DynamicArrayNumbers *a, DynamicArrayNumbers *b){
  bool equal;
  double i;

  equal = true;
  if(a->length == b->length){
    for(i = 0.0; i < a->length && equal; i = i + 1.0){
      if(a->array->at(i) != b->array->at(i)){
        equal = false;
      }
    }
  }else{
    equal = false;
  }

  return equal;
}
LinkedListNumbers *DynamicArrayNumbersToLinkedList(DynamicArrayNumbers *da){
  LinkedListNumbers *ll;
  double i;

  ll = CreateLinkedListNumbers();

  for(i = 0.0; i < da->length; i = i + 1.0){
    LinkedListAddNumber(ll, da->array->at(i));
  }

  return ll;
}
DynamicArrayNumbers *LinkedListToDynamicArrayNumbers(LinkedListNumbers *ll){
  DynamicArrayNumbers *da;
  double i;
  LinkedListNodeNumbers *node;

  node = ll->first;

  da = new DynamicArrayNumbers();
  da->length = LinkedListNumbersLength(ll);

  da->array = new vector<double> (da->length);

  for(i = 0.0; i < da->length; i = i + 1.0){
    da->array->at(i) = node->value;
    node = node->next;
  }

  return da;
}
vector<wchar_t> *AddCharacter(vector<wchar_t> *list, wchar_t a){
  vector<wchar_t> *newlist;
  double i;

  newlist = new vector<wchar_t> (list->size() + 1.0);
  for(i = 0.0; i < list->size(); i = i + 1.0){
    newlist->at(i) = list->at(i);
  }
  newlist->at(list->size()) = a;
		
  delete list;
		
  return newlist;
}
void AddCharacterRef(StringReference *list, wchar_t i){
  list->string = AddCharacter(list->string, i);
}
vector<wchar_t> *RemoveCharacter(vector<wchar_t> *list, double n){
  vector<wchar_t> *newlist;
  double i;

  newlist = new vector<wchar_t> (list->size() - 1.0);

  if(n >= 0.0 && n < list->size()){
    for(i = 0.0; i < list->size(); i = i + 1.0){
      if(i < n){
        newlist->at(i) = list->at(i);
      }
      if(i > n){
        newlist->at(i - 1.0) = list->at(i);
      }
    }

    delete list;
  }else{
    delete newlist;
  }

  return newlist;
}
wchar_t GetCharacterRef(StringReference *list, double i){
  return list->string->at(i);
}
void RemoveCharacterRef(StringReference *list, double i){
  list->string = RemoveCharacter(list->string, i);
}
wchar_t charToLowerCase(wchar_t character){
  wchar_t toReturn;

  toReturn = character;
  if(character == 'A'){
    toReturn = 'a';
  }else if(character == 'B'){
    toReturn = 'b';
  }else if(character == 'C'){
    toReturn = 'c';
  }else if(character == 'D'){
    toReturn = 'd';
  }else if(character == 'E'){
    toReturn = 'e';
  }else if(character == 'F'){
    toReturn = 'f';
  }else if(character == 'G'){
    toReturn = 'g';
  }else if(character == 'H'){
    toReturn = 'h';
  }else if(character == 'I'){
    toReturn = 'i';
  }else if(character == 'J'){
    toReturn = 'j';
  }else if(character == 'K'){
    toReturn = 'k';
  }else if(character == 'L'){
    toReturn = 'l';
  }else if(character == 'M'){
    toReturn = 'm';
  }else if(character == 'N'){
    toReturn = 'n';
  }else if(character == 'O'){
    toReturn = 'o';
  }else if(character == 'P'){
    toReturn = 'p';
  }else if(character == 'Q'){
    toReturn = 'q';
  }else if(character == 'R'){
    toReturn = 'r';
  }else if(character == 'S'){
    toReturn = 's';
  }else if(character == 'T'){
    toReturn = 't';
  }else if(character == 'U'){
    toReturn = 'u';
  }else if(character == 'V'){
    toReturn = 'v';
  }else if(character == 'W'){
    toReturn = 'w';
  }else if(character == 'X'){
    toReturn = 'x';
  }else if(character == 'Y'){
    toReturn = 'y';
  }else if(character == 'Z'){
    toReturn = 'z';
  }

  return toReturn;
}
wchar_t charToUpperCase(wchar_t character){
  wchar_t toReturn;

  toReturn = character;
  if(character == 'a'){
    toReturn = 'A';
  }else if(character == 'b'){
    toReturn = 'B';
  }else if(character == 'c'){
    toReturn = 'C';
  }else if(character == 'd'){
    toReturn = 'D';
  }else if(character == 'e'){
    toReturn = 'E';
  }else if(character == 'f'){
    toReturn = 'F';
  }else if(character == 'g'){
    toReturn = 'G';
  }else if(character == 'h'){
    toReturn = 'H';
  }else if(character == 'i'){
    toReturn = 'I';
  }else if(character == 'j'){
    toReturn = 'J';
  }else if(character == 'k'){
    toReturn = 'K';
  }else if(character == 'l'){
    toReturn = 'L';
  }else if(character == 'm'){
    toReturn = 'M';
  }else if(character == 'n'){
    toReturn = 'N';
  }else if(character == 'o'){
    toReturn = 'O';
  }else if(character == 'p'){
    toReturn = 'P';
  }else if(character == 'q'){
    toReturn = 'Q';
  }else if(character == 'r'){
    toReturn = 'R';
  }else if(character == 's'){
    toReturn = 'S';
  }else if(character == 't'){
    toReturn = 'T';
  }else if(character == 'u'){
    toReturn = 'U';
  }else if(character == 'v'){
    toReturn = 'V';
  }else if(character == 'w'){
    toReturn = 'W';
  }else if(character == 'x'){
    toReturn = 'X';
  }else if(character == 'y'){
    toReturn = 'Y';
  }else if(character == 'z'){
    toReturn = 'Z';
  }

  return toReturn;
}
bool charIsUpperCase(wchar_t character){
  bool isUpper;

  isUpper = false;
  if(character == 'A'){
    isUpper = true;
  }else if(character == 'B'){
    isUpper = true;
  }else if(character == 'C'){
    isUpper = true;
  }else if(character == 'D'){
    isUpper = true;
  }else if(character == 'E'){
    isUpper = true;
  }else if(character == 'F'){
    isUpper = true;
  }else if(character == 'G'){
    isUpper = true;
  }else if(character == 'H'){
    isUpper = true;
  }else if(character == 'I'){
    isUpper = true;
  }else if(character == 'J'){
    isUpper = true;
  }else if(character == 'K'){
    isUpper = true;
  }else if(character == 'L'){
    isUpper = true;
  }else if(character == 'M'){
    isUpper = true;
  }else if(character == 'N'){
    isUpper = true;
  }else if(character == 'O'){
    isUpper = true;
  }else if(character == 'P'){
    isUpper = true;
  }else if(character == 'Q'){
    isUpper = true;
  }else if(character == 'R'){
    isUpper = true;
  }else if(character == 'S'){
    isUpper = true;
  }else if(character == 'T'){
    isUpper = true;
  }else if(character == 'U'){
    isUpper = true;
  }else if(character == 'V'){
    isUpper = true;
  }else if(character == 'W'){
    isUpper = true;
  }else if(character == 'X'){
    isUpper = true;
  }else if(character == 'Y'){
    isUpper = true;
  }else if(character == 'Z'){
    isUpper = true;
  }

  return isUpper;
}
bool charIsLowerCase(wchar_t character){
  bool isLower;

  isLower = false;
  if(character == 'a'){
    isLower = true;
  }else if(character == 'b'){
    isLower = true;
  }else if(character == 'c'){
    isLower = true;
  }else if(character == 'd'){
    isLower = true;
  }else if(character == 'e'){
    isLower = true;
  }else if(character == 'f'){
    isLower = true;
  }else if(character == 'g'){
    isLower = true;
  }else if(character == 'h'){
    isLower = true;
  }else if(character == 'i'){
    isLower = true;
  }else if(character == 'j'){
    isLower = true;
  }else if(character == 'k'){
    isLower = true;
  }else if(character == 'l'){
    isLower = true;
  }else if(character == 'm'){
    isLower = true;
  }else if(character == 'n'){
    isLower = true;
  }else if(character == 'o'){
    isLower = true;
  }else if(character == 'p'){
    isLower = true;
  }else if(character == 'q'){
    isLower = true;
  }else if(character == 'r'){
    isLower = true;
  }else if(character == 's'){
    isLower = true;
  }else if(character == 't'){
    isLower = true;
  }else if(character == 'u'){
    isLower = true;
  }else if(character == 'v'){
    isLower = true;
  }else if(character == 'w'){
    isLower = true;
  }else if(character == 'x'){
    isLower = true;
  }else if(character == 'y'){
    isLower = true;
  }else if(character == 'z'){
    isLower = true;
  }

  return isLower;
}
bool charIsLetter(wchar_t character){
  return charIsUpperCase(character) || charIsLowerCase(character);
}
bool charIsNumber(wchar_t character){
  bool isNumberx;

  isNumberx = false;
  if(character == '0'){
    isNumberx = true;
  }else if(character == '1'){
    isNumberx = true;
  }else if(character == '2'){
    isNumberx = true;
  }else if(character == '3'){
    isNumberx = true;
  }else if(character == '4'){
    isNumberx = true;
  }else if(character == '5'){
    isNumberx = true;
  }else if(character == '6'){
    isNumberx = true;
  }else if(character == '7'){
    isNumberx = true;
  }else if(character == '8'){
    isNumberx = true;
  }else if(character == '9'){
    isNumberx = true;
  }

  return isNumberx;
}
bool charIsWhiteSpace(wchar_t character){
  bool isWhiteSpacex;

  isWhiteSpacex = false;
  if(character == ' '){
    isWhiteSpacex = true;
  }else if(character == '\t'){
    isWhiteSpacex = true;
  }else if(character == '\n'){
    isWhiteSpacex = true;
  }else if(character == '\r'){
    isWhiteSpacex = true;
  }

  return isWhiteSpacex;
}
bool charIsSymbol(wchar_t character){
  bool isSymbolx;

  isSymbolx = false;
  if(character == '!'){
    isSymbolx = true;
  }else if(character == '\"'){
    isSymbolx = true;
  }else if(character == '#'){
    isSymbolx = true;
  }else if(character == '$'){
    isSymbolx = true;
  }else if(character == '%'){
    isSymbolx = true;
  }else if(character == '&'){
    isSymbolx = true;
  }else if(character == '\''){
    isSymbolx = true;
  }else if(character == '('){
    isSymbolx = true;
  }else if(character == ')'){
    isSymbolx = true;
  }else if(character == '*'){
    isSymbolx = true;
  }else if(character == '+'){
    isSymbolx = true;
  }else if(character == ','){
    isSymbolx = true;
  }else if(character == '-'){
    isSymbolx = true;
  }else if(character == '.'){
    isSymbolx = true;
  }else if(character == '/'){
    isSymbolx = true;
  }else if(character == ':'){
    isSymbolx = true;
  }else if(character == ';'){
    isSymbolx = true;
  }else if(character == '<'){
    isSymbolx = true;
  }else if(character == '='){
    isSymbolx = true;
  }else if(character == '>'){
    isSymbolx = true;
  }else if(character == '?'){
    isSymbolx = true;
  }else if(character == '@'){
    isSymbolx = true;
  }else if(character == '['){
    isSymbolx = true;
  }else if(character == '\\'){
    isSymbolx = true;
  }else if(character == ']'){
    isSymbolx = true;
  }else if(character == '^'){
    isSymbolx = true;
  }else if(character == '_'){
    isSymbolx = true;
  }else if(character == '`'){
    isSymbolx = true;
  }else if(character == '{'){
    isSymbolx = true;
  }else if(character == '|'){
    isSymbolx = true;
  }else if(character == '}'){
    isSymbolx = true;
  }else if(character == '~'){
    isSymbolx = true;
  }

  return isSymbolx;
}
bool charCharacterIsBefore(wchar_t a, wchar_t b){
  double ad, bd;

  ad = a;
  bd = b;

  return ad < bd;
}
double And4Byte(double n1, double n2){
    if((double)n1 >= 0.0 && (double)n1 <= (double)0xFFFFFFFFUL && (double)n2 >= 0.0 && (double)n2 <= (double)0xFFFFFFFFUL){
      return (unsigned long)n1 & (unsigned long)n2;
    }else{
      return 0.0;
    }
}
double And2Byte(double n1, double n2){
    if((double)n1 >= 0.0 && (double)n1 <= (double)0xFFFFUL && (double)n2 >= 0.0 && (double)n2 <= (double)0xFFFFUL){
      return (unsigned long)n1 & (unsigned long)n2;
    }else{
      return 0.0;
    }
}
double AndByte(double n1, double n2){
    if((double)n1 >= 0.0 && (double)n1 <= (double)0xFFUL && (double)n2 >= 0.0 && (double)n2 <= (double)0xFFUL){
      return (unsigned long)n1 & (unsigned long)n2;
    }else{
      return 0.0;
    }
}
double AndBytes(double n1, double n2, double bytes){
  double byteVal, result, i;

  byteVal = 1.0;
  result = 0.0;

  if(n1 >= 0.0 && n1 < pow(2.0, bytes*8.0) && n2 >= 0.0 && n2 < pow(2.0, bytes*8.0)){
    n1 = Truncate(n1);
    n2 = Truncate(n2);
    bytes = Truncate(bytes);

    for(i = 0.0; i < bytes*8.0; i = i + 1.0){
      if(fmod(n1, 2.0) == 1.0 && fmod(n2, 2.0) == 1.0){
        result = result + byteVal;
      }
      n1 = floor(n1/2.0);
      n2 = floor(n2/2.0);
      byteVal = byteVal*2.0;
    }
  }

  return result;
}
double Or4Byte(double n1, double n2){
    if((double)n1 >= 0.0 && (double)n1 <= (double)0xFFFFFFFFUL && (double)n2 >= 0.0 && (double)n2 <= (double)0xFFFFFFFFUL){
      return (unsigned long)n1 | (unsigned long)n2;
    }else{
      return 0.0;
    }
}
double Or2Byte(double n1, double n2){
    if((double)n1 >= 0.0 && (double)n1 <= (double)0xFFFFUL && (double)n2 >= 0.0 && (double)n2 <= (double)0xFFFFUL){
      return (unsigned long)n1 | (unsigned long)n2;
    }else{
      return 0.0;
    }
}
double OrByte(double n1, double n2){
    if((double)n1 >= 0.0 && (double)n1 <= (double)0xFFUL && (double)n2 >= 0.0 && (double)n2 <= (double)0xFFUL){
      return (unsigned long)n1 | (unsigned long)n2;
    }else{
      return 0.0;
    }
}
double OrBytes(double n1, double n2, double bytes){
  double byteVal, result, i;

  byteVal = 1.0;
  result = 0.0;

  if(n1 >= 0.0 && n1 < pow(2.0, bytes*8.0) && n2 >= 0.0 && n2 < pow(2.0, bytes*8.0)){
    n1 = Truncate(n1);
    n2 = Truncate(n2);
    bytes = Truncate(bytes);

    for(i = 0.0; i < bytes*8.0; i = i + 1.0){
      if(fmod(n1, 2.0) == 1.0 || fmod(n2, 2.0) == 1.0){
        result = result + byteVal;
      }
      n1 = floor(n1/2.0);
      n2 = floor(n2/2.0);
      byteVal = byteVal*2.0;
    }
  }

  return result;
}
double Xor4Byte(double n1, double n2){
    if((double)n1 >= 0.0 && (double)n1 <= (double)0xFFFFFFFFUL && (double)n2 >= 0.0 && (double)n2 <= (double)0xFFFFFFFFUL){
      return (unsigned long)n1 ^ (unsigned long)n2;
    }else{
      return 0.0;
    }
}
double Xor2Byte(double n1, double n2){
    if((double)n1 >= 0.0 && (double)n1 <= (double)0xFFFFUL && (double)n2 >= 0.0 && (double)n2 <= (double)0xFFFFUL){
      return (unsigned long)n1 ^ (unsigned long)n2;
    }else{
      return 0.0;
    }
}
double XorByte(double n1, double n2){
    if((double)n1 >= 0.0 && (double)n1 <= (double)0xFFUL && (double)n2 >= 0.0 && (double)n2 <= (double)0xFFUL){
      return (unsigned long)n1 ^ (unsigned long)n2;
    }else{
      return 0.0;
    }
}
double XorBytes(double n1, double n2, double bytes){
  double byteVal, result, i;

  byteVal = 1.0;
  result = 0.0;

  if(n1 >= 0.0 && n1 < pow(2.0, bytes*8.0) && n2 >= 0.0 && n2 < pow(2.0, bytes*8.0)){
    n1 = Truncate(n1);
    n2 = Truncate(n2);
    bytes = Truncate(bytes);

    for(i = 0.0; i < bytes*8.0; i = i + 1.0){
      if(fmod(n1, 2.0) != fmod(n2, 2.0)){
        result = result + byteVal;
      }
      n1 = floor(n1/2.0);
      n2 = floor(n2/2.0);
      byteVal = byteVal*2.0;
    }
  }

  return result;
}
double Not4Byte(double b){
    if((double)b >= 0.0 && (double)b <= (double)0xFFFFFFFFUL){
      return ~(unsigned long)b & 0xFFFFFFFFUL;
    }else{
      return 0.0;
    }
}
double Not2Byte(double b){
    if((double)b >= 0.0 && (double)b <= (double)0xFFFFUL){
      return ~(unsigned long)b & 0xFFFFUL;
    }else{
      return 0.0;
    }
}
double NotByte(double b){
    if((double)b >= 0.0 && (double)b <= (double)0xFFUL){
      return ~(unsigned long)b & 0xFFUL;
    }else{
      return 0.0;
    }
}
double NotBytes(double b, double length){
  double result;

  result = 0.0;

  if(b >= 0.0 && b < pow(2.0, length*8.0)){
    b = Truncate(b);
    length = Truncate(length);

    result = pow(2.0, length*8.0) - b - 1.0;
  }

  return result;
}
double ShiftLeft4Byte(double b, double amount){
    if((double)b >= 0.0 && (double)b <= (double)0xFFFFFFFF && (double)amount >= 0.0 && (double)amount < (double)32){
      return (unsigned long)b << (unsigned long)amount;
    }else{
      return 0.0;
    }
}
double ShiftLeft2Byte(double b, double amount){
    if((double)b >= 0.0 && (double)b <= (double)0xFFFF && (double)amount >= 0.0 && (double)amount < (double)16){
      return (unsigned long)b << (unsigned long)amount;
    }else{
      return 0.0;
    }
}
double ShiftLeftByte(double b, double amount){
    if((double)b >= 0.0 && (double)b <= (double)0xFF && (double)amount >= 0.0 && (double)amount < (double)8){
      return (unsigned long)b << (unsigned long)amount;
    }else{
      return 0.0;
    }
}
double ShiftLeftBytes(double b, double amount, double length){
  double result;

  result = 0.0;

  if(b >= 0.0 && b < pow(2.0, length*8.0) && amount >= 0.0 && amount <= length*8.0){
    b = Truncate(b);
    amount = Truncate(amount);

    result = b*pow(2.0, amount);
  }

  return result;
}
double ShiftRight4Byte(double b, double amount){
    if((double)b >= 0.0 && (double)b <= (double)0xFFFFFFFF && (double)amount >= 0.0 && (double)amount < (double)32){
      return (unsigned long)b >> (unsigned long)amount;
    }else{
      return 0.0;
    }
}
double ShiftRight2Byte(double b, double amount){
    if((double)b >= 0.0 && (double)b <= (double)0xFFFF && (double)amount >= 0.0 && (double)amount < (double)16){
      return (unsigned long)b >> (unsigned long)amount;
    }else{
      return 0.0;
    }
}
double ShiftRightByte(double b, double amount){
    if((double)b >= 0.0 && (double)b <= (double)0xFF && (double)amount >= 0.0 && (double)amount < (double)8){
      return (unsigned long)b >> (unsigned long)amount;
    }else{
      return 0.0;
    }
}
double ShiftRightBytes(double b, double amount, double length){
  double result;

  result = 0.0;

  if(b >= 0.0 && b < pow(2.0, length*8.0) && amount >= 0.0 && amount <= length*8.0){
    b = Truncate(b);
    amount = Truncate(amount);

    result = Truncate(b/pow(2.0, amount));
  }

  return result;
}
double ReadNextBit(vector<double> *data, NumberReference *nextbit){
  double bytenr, bitnumber, bit, b;

  bytenr = floor(nextbit->numberValue/8.0);
  bitnumber = fmod(nextbit->numberValue, 8.0);

  b = data->at(bytenr);

  bit = fmod(floor(b/pow(2.0, bitnumber)), 2.0);

  nextbit->numberValue = nextbit->numberValue + 1.0;

  return bit;
}
double BitExtract(double b, double fromInc, double toInc){
  return fmod(floor(b/pow(2.0, fromInc)), pow(2.0, toInc + 1.0 - fromInc));
}
double ReadBitRange(vector<double> *data, NumberReference *nextbit, double length){
  double startbyte, endbyte;
  double startbit, endbit;
  double number, i;

  number = 0.0;

  startbyte = floor(nextbit->numberValue/8.0);
  endbyte = floor((nextbit->numberValue + length)/8.0);

  startbit = fmod(nextbit->numberValue, 8.0);
  endbit = fmod(nextbit->numberValue + length - 1.0, 8.0);

  if(startbyte == endbyte){
    number = BitExtract(data->at(startbyte), startbit, endbit);
  }

  nextbit->numberValue = nextbit->numberValue + length;

  return number;
}
void SkipToBoundary(NumberReference *nextbit){
  double skip;

  skip = 8.0 - fmod(nextbit->numberValue, 8.0);
  nextbit->numberValue = nextbit->numberValue + skip;
}
double ReadNextByteBoundary(vector<double> *data, NumberReference *nextbit){
  double bytenr, b;

  bytenr = floor(nextbit->numberValue/8.0);
  b = data->at(bytenr);
  nextbit->numberValue = nextbit->numberValue + 8.0;

  return b;
}
double Read2bytesByteBoundary(vector<double> *data, NumberReference *nextbit){
  double r;

  r = 0.0;
  r = r + pow(2.0, 8.0)*ReadNextByteBoundary(data, nextbit);
  r = r + ReadNextByteBoundary(data, nextbit);

  return r;
}
double ComputeAdler32(vector<double> *data){
  double a, b, m, i;

  a = 1.0;
  b = 0.0;
  m = 65521.0;

  for(i = 0.0; i < data->size(); i = i + 1.0){
    a = fmod(a + data->at(i), m);
    b = fmod(b + a, m);
  }

  return b*pow(2.0, 16.0) + a;
}
vector<double> *DeflateDataStaticHuffman(vector<double> *data, double level){
  vector<double> *bytes;
  NumberReference *currentBit;
  double i;
  NumberArrayReference *copy;
  NumberReference *code, *length, *compressedCode, *lengthAdditionLength, *distanceCode;
  NumberReference *distanceReference, *lengthReference, *lengthAddition;
  NumberReference *distanceAdditionReference, *distanceAdditionLengthReference;
  vector<double> *bitReverseLookupTable;
  BooleanReference *match;

  code = CreateNumberReference(0.0);
  length = CreateNumberReference(0.0);
  compressedCode = CreateNumberReference(0.0);
  lengthAdditionLength = CreateNumberReference(0.0);
  distanceCode = CreateNumberReference(0.0);
  distanceReference = CreateNumberReference(0.0);
  lengthReference = CreateNumberReference(0.0);
  lengthAddition = CreateNumberReference(0.0);
  distanceAdditionReference = CreateNumberReference(0.0);
  distanceAdditionLengthReference = CreateNumberReference(0.0);
  match = new BooleanReference();

  bytes = new vector<double> (fmax(data->size()*2.0, 100.0));
  aFillNumberArray(bytes, 0.0);
  currentBit = CreateNumberReference(0.0);

  bitReverseLookupTable = GenerateBitReverseLookupTable(9.0);

  /* Final block */
  AppendBitsToBytesRight(bytes, currentBit, 1.0, 1.0);
  /* Fixed code */
  AppendBitsToBytesRight(bytes, currentBit, 1.0, 2.0);

  for(i = 0.0; i < data->size(); ){
    FindMatch(data, i, distanceReference, lengthReference, match, level);

    if(match->booleanValue){
      GetDeflateLengthCode(lengthReference->numberValue, compressedCode, lengthAddition, lengthAdditionLength);
      GetDeflateDistanceCode(distanceReference->numberValue, distanceCode, distanceAdditionReference, distanceAdditionLengthReference, bitReverseLookupTable);
    }

    if( !match->booleanValue ){
      GetDeflateStaticHuffmanCode(data->at(i), code, length, bitReverseLookupTable);
      AppendBitsToBytesRight(bytes, currentBit, code->numberValue, length->numberValue);
      i = i + 1.0;
    }else{
      GetDeflateStaticHuffmanCode(compressedCode->numberValue, code, length, bitReverseLookupTable);
      AppendBitsToBytesRight(bytes, currentBit, code->numberValue, length->numberValue);
      AppendBitsToBytesRight(bytes, currentBit, lengthAddition->numberValue, lengthAdditionLength->numberValue);
      AppendBitsToBytesRight(bytes, currentBit, distanceCode->numberValue, 5.0);
      AppendBitsToBytesRight(bytes, currentBit, distanceAdditionReference->numberValue, distanceAdditionLengthReference->numberValue);
      i = i + lengthReference->numberValue;
    }
  }

  /* Stop symbol */
  GetDeflateStaticHuffmanCode(256.0, code, length, bitReverseLookupTable);
  AppendBitsToBytesRight(bytes, currentBit, code->numberValue, length->numberValue);

  copy = new NumberArrayReference();
  aCopyNumberArrayRange(bytes, 0.0, ceil(currentBit->numberValue/8.0), copy);
  delete bytes;
  bytes = copy->numberArray;

  return bytes;
}
void FindMatch(vector<double> *data, double pos, NumberReference *distanceReference, NumberReference *lengthReference, BooleanReference *match, double level){
  double i, j;
  double deflateMinMength, deflateMaxLength, deflateMaxDistance;
  double longest, maxLength, distanceForMax;
  double startDistance, matchLength;
  bool done;

  deflateMinMength = 3.0;
  deflateMaxLength = 258.0;

  longest = fmin(pos - 1.0, deflateMaxLength);
  longest = fmin(data->size() - pos, longest);

  deflateMaxDistance = floor(32768.0/10.0*level);

  startDistance = fmin(pos, deflateMaxDistance);

  if(longest >= deflateMinMength){
    maxLength = 0.0;
    distanceForMax = 0.0;

    for(i = pos - 1.0; i >= pos - startDistance && maxLength != longest; i = i - 1.0){
      matchLength = 0.0;
      done = false;
      for(j = 0.0; j < longest &&  !done ; j = j + 1.0){
        if(data->at(i + j) == data->at(pos + j)){
          matchLength = matchLength + 1.0;
        }else{
          done = true;
        }
      }

      if(matchLength >= deflateMinMength && matchLength > maxLength){
        maxLength = matchLength;
        distanceForMax = pos - i;
      }
    }

    if(maxLength >= deflateMinMength){
      match->booleanValue = true;
      lengthReference->numberValue = maxLength;
      distanceReference->numberValue = distanceForMax;
    }else{
      match->booleanValue = false;
    }
  }else{
    match->booleanValue = false;
  }
}
vector<double> *GenerateBitReverseLookupTable(double bits){
  vector<double> *table;
  double i;

  table = new vector<double> (pow(2.0, bits));

  for(i = 0.0; i < table->size(); i = i + 1.0){
    table->at(i) = ReverseBits(i, 32.0);
  }

  return table;
}
double ReverseBits(double x, double bits){
  double b, bit, i;

  b = 0.0;

  for(i = 0.0; i < bits; i = i + 1.0){
    b = ShiftLeft4Byte(b, 1.0);
    bit = And4Byte(x, 1.0);
    b = Or4Byte(b, bit);
    x = ShiftRight4Byte(x, 1.0);
  }

  return b;
}
vector<double> *DeflateDataNoCompression(vector<double> *data){
  vector<double> *deflated;
  NumberReference *position;
  double block, i, blocks, blocklength, maxblocksize;

  maxblocksize = pow(2.0, 16.0) - 1.0;
  blocks = ceil(data->size()/maxblocksize);

  position = CreateNumberReference(0.0);

  deflated = new vector<double> ((1.0 + 4.0)*blocks + data->size());

  for(block = 0.0; block < blocks; block = block + 1.0){
    if(block + 1.0 == blocks){
      WriteByte(deflated, 1.0, position);
    }else{
      WriteByte(deflated, 0.0, position);
    }
    blocklength = fmin(data->size() - block*maxblocksize, maxblocksize);
    Write2BytesLE(deflated, blocklength, position);
    Write2BytesLE(deflated, Not2Byte(blocklength), position);

    for(i = 0.0; i < blocklength; i = i + 1.0){
      WriteByte(deflated, data->at(block*maxblocksize + i), position);
    }
  }

  return deflated;
}
void GetDeflateStaticHuffmanCode(double b, NumberReference *code, NumberReference *length, vector<double> *bitReverseLookupTable){
  double reversed;

  if(b >= 0.0 && b <= 143.0){
    code->numberValue = 48.0 + b;
    length->numberValue = 8.0;
  }else if(b >= 144.0 && b <= 255.0){
    code->numberValue = b - 144.0 + 400.0;
    length->numberValue = 9.0;
  }else if(b >= 256.0 && b <= 279.0){
    code->numberValue = b - 256.0 + 0.0;
    length->numberValue = 7.0;
  }else if(b >= 280.0 && b <= 287.0){
    code->numberValue = b - 280.0 + 192.0;
    length->numberValue = 8.0;
  }

  reversed = bitReverseLookupTable->at(code->numberValue);
  code->numberValue = ShiftRight4Byte(reversed, 32.0 - length->numberValue);
}
void GetDeflateLengthCode(double length, NumberReference *code, NumberReference *lengthAddition, NumberReference *lengthAdditionLength){
  if(length >= 3.0 && length <= 10.0){
    code->numberValue = 257.0 + length - 3.0;
    lengthAdditionLength->numberValue = 0.0;
  }else if(length >= 11.0 && length <= 18.0){
    code->numberValue = 265.0 + floor((length - 11.0)/2.0);
    lengthAddition->numberValue = floor(fmod(length - 11.0, 2.0));
    lengthAdditionLength->numberValue = 1.0;
  }else if(length >= 19.0 && length <= 34.0){
    code->numberValue = 269.0 + floor((length - 19.0)/4.0);
    lengthAddition->numberValue = floor(fmod(length - 19.0, 4.0));
    lengthAdditionLength->numberValue = 2.0;
  }else if(length >= 35.0 && length <= 66.0){
    code->numberValue = 273.0 + floor((length - 35.0)/8.0);
    lengthAddition->numberValue = floor(fmod(length - 35.0, 8.0));
    lengthAdditionLength->numberValue = 3.0;
  }else if(length >= 67.0 && length <= 130.0){
    code->numberValue = 277.0 + floor((length - 67.0)/16.0);
    lengthAddition->numberValue = floor(fmod(length - 67.0, 16.0));
    lengthAdditionLength->numberValue = 4.0;
  }else if(length >= 131.0 && length <= 257.0){
    code->numberValue = 281.0 + floor((length - 131.0)/32.0);
    lengthAddition->numberValue = floor(fmod(length - 131.0, 32.0));
    lengthAdditionLength->numberValue = 5.0;
  }else if(length == 258.0){
    code->numberValue = 285.0;
    lengthAdditionLength->numberValue = 0.0;
  }
}
void GetDeflateDistanceCode(double distance, NumberReference *code, NumberReference *distanceAdditionReference, NumberReference *distanceAdditionLengthReference, vector<double> *bitReverseLookupTable){
  double reversed;

  if(distance >= 1.0 && distance <= 4.0){
    code->numberValue = distance - 1.0;
    distanceAdditionLengthReference->numberValue = 0.0;
  }else if(distance >= 5.0 && distance <= 8.0){
    code->numberValue = 4.0 + floor((distance - 5.0)/2.0);
    distanceAdditionReference->numberValue = floor(fmod(distance - 5.0, 2.0));
    distanceAdditionLengthReference->numberValue = 1.0;
  }else if(distance >= 9.0 && distance <= 16.0){
    code->numberValue = 6.0 + floor((distance - 9.0)/4.0);
    distanceAdditionReference->numberValue = floor(fmod(distance - 9.0, 4.0));
    distanceAdditionLengthReference->numberValue = 2.0;
  }else if(distance >= 17.0 && distance <= 32.0){
    code->numberValue = 8.0 + floor((distance - 17.0)/8.0);
    distanceAdditionReference->numberValue = floor(fmod(distance - 17.0, 8.0));
    distanceAdditionLengthReference->numberValue = 3.0;
  }else if(distance >= 33.0 && distance <= 64.0){
    code->numberValue = 10.0 + floor((distance - 33.0)/16.0);
    distanceAdditionReference->numberValue = floor(fmod(distance - 33.0, 16.0));
    distanceAdditionLengthReference->numberValue = 4.0;
  }else if(distance >= 65.0 && distance <= 128.0){
    code->numberValue = 12.0 + floor((distance - 65.0)/32.0);
    distanceAdditionReference->numberValue = floor(fmod(distance - 65.0, 32.0));
    distanceAdditionLengthReference->numberValue = 5.0;
  }else if(distance >= 129.0 && distance <= 256.0){
    code->numberValue = 14.0 + floor((distance - 129.0)/64.0);
    distanceAdditionReference->numberValue = floor(fmod(distance - 129.0, 64.0));
    distanceAdditionLengthReference->numberValue = 6.0;
  }else if(distance >= 257.0 && distance <= 512.0){
    code->numberValue = 16.0 + floor((distance - 257.0)/128.0);
    distanceAdditionReference->numberValue = floor(fmod(distance - 257.0, 128.0));
    distanceAdditionLengthReference->numberValue = 7.0;
  }else if(distance >= 513.0 && distance <= 1024.0){
    code->numberValue = 18.0 + floor((distance - 513.0)/256.0);
    distanceAdditionReference->numberValue = floor(fmod(distance - 513.0, 256.0));
    distanceAdditionLengthReference->numberValue = 8.0;
  }else if(distance >= 1025.0 && distance <= 2048.0){
    code->numberValue = 20.0 + floor((distance - 1025.0)/pow(2.0, 9.0));
    distanceAdditionReference->numberValue = floor(fmod(distance - 1025.0, pow(2.0, 9.0)));
    distanceAdditionLengthReference->numberValue = 9.0;
  }else if(distance >= 2049.0 && distance <= 4096.0){
    code->numberValue = 22.0 + floor((distance - 2049.0)/pow(2.0, 10.0));
    distanceAdditionReference->numberValue = floor(fmod(distance - 2049.0, pow(2.0, 10.0)));
    distanceAdditionLengthReference->numberValue = 10.0;
  }else if(distance >= 4097.0 && distance <= 8192.0){
    code->numberValue = 24.0 + floor((distance - 4097.0)/pow(2.0, 11.0));
    distanceAdditionReference->numberValue = floor(fmod(distance - 4097.0, pow(2.0, 11.0)));
    distanceAdditionLengthReference->numberValue = 11.0;
  }else if(distance >= 8193.0 && distance <= 16384.0){
    code->numberValue = 26.0 + floor((distance - 8193.0)/pow(2.0, 12.0));
    distanceAdditionReference->numberValue = floor(fmod(distance - 8193.0, pow(2.0, 12.0)));
    distanceAdditionLengthReference->numberValue = 12.0;
  }else if(distance >= 16385.0 && distance <= 32768.0){
    code->numberValue = 28.0 + floor((distance - 16385.0)/pow(2.0, 13.0));
    distanceAdditionReference->numberValue = floor(fmod(distance - 16385.0, pow(2.0, 13.0)));
    distanceAdditionLengthReference->numberValue = 13.0;
  }

  reversed = bitReverseLookupTable->at(code->numberValue);
  code->numberValue = ShiftRight4Byte(reversed, 32.0 - 5.0);
}
void AppendBitsToBytesLeft(vector<double> *bytes, NumberReference *nextbit, double data, double length){
  double bytePos, bitPos, segment, part, remove;

  for(; length > 0.0; ){
    bytePos = Truncate(nextbit->numberValue/8.0);
    bitPos = fmod(nextbit->numberValue, 8.0);

    if(length < 8.0 - bitPos){
      part = ShiftLeft4Byte(data, 8.0 - bitPos - length);

      bytes->at(bytePos) = Or4Byte(bytes->at(bytePos), part);

      nextbit->numberValue = nextbit->numberValue + length;

      length = 0.0;
    }else{
      segment = 8.0 - bitPos;

      part = ShiftRight4Byte(data, length - segment);
      bytes->at(bytePos) = Or4Byte(bytes->at(bytePos), part);
      nextbit->numberValue = nextbit->numberValue + segment;

      remove = ShiftLeft4Byte(part, length - segment);
      data = Xor4Byte(data, remove);

      length = length - segment;
    }
  }
}
void AppendBitsToBytesRight(vector<double> *bytes, NumberReference *nextbit, double data, double length){
  double bytePos, bitPos, segment, part;
  double mask;

  for(; length > 0.0; ){
    bytePos = Truncate(nextbit->numberValue/8.0);
    bitPos = fmod(nextbit->numberValue, 8.0);

    if(length < 8.0 - bitPos){
      part = ShiftLeft4Byte(data, bitPos);

      bytes->at(bytePos) = Or4Byte(bytes->at(bytePos), part);

      nextbit->numberValue = nextbit->numberValue + length;

      length = 0.0;
    }else{
      segment = 8.0 - bitPos;

      mask = 1.0;
      mask = ShiftLeft4Byte(mask, segment);
      mask = mask - 1.0;

      part = And4Byte(mask, data);
      part = ShiftLeft4Byte(part, bitPos);
      bytes->at(bytePos) = Or4Byte(bytes->at(bytePos), part);
      nextbit->numberValue = nextbit->numberValue + segment;

      data = ShiftRight4Byte(data, segment);

      length = length - segment;
    }
  }
}
