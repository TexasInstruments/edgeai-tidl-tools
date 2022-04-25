// Downloaded from https://repo.progsbase.com - Code Developed Using progsbase.

#include <cmath>
#include <cstring>
#include <vector>
#include <cwchar>

#define toVector(s) (new std::vector<wchar_t> ((s), (s) + wcslen(s)))

struct RGBABitmapImageReference;

struct Rectangle;

struct ScatterPlotSeries;

struct ScatterPlotSettings;

struct BarPlotSeries;

struct BarPlotSettings;

struct RGBA;

struct RGBABitmap;

struct RGBABitmapImage;

struct BooleanArrayReference;

struct BooleanReference;

struct CharacterReference;

struct NumberArrayReference;

struct NumberReference;

struct StringArrayReference;

struct StringReference;

struct Chunk;

struct IHDR;

struct PHYS;

struct PNGImage;

struct ZLIBStruct;

struct LinkedListNodeStrings;

struct LinkedListStrings;

struct LinkedListNodeNumbers;

struct LinkedListNumbers;

struct LinkedListCharacters;

struct LinkedListNodeCharacters;

struct DynamicArrayNumbers;

struct RGBABitmapImageReference{
  RGBABitmapImage *image;
};

struct Rectangle{
  double x1;
  double x2;
  double y1;
  double y2;
};

struct ScatterPlotSeries{
  bool linearInterpolation;
  std::vector<wchar_t> *pointType;
  std::vector<wchar_t> *lineType;
  double lineThickness;
  std::vector<double> *xs;
  std::vector<double> *ys;
  RGBA *color;
};

struct ScatterPlotSettings{
  std::vector<ScatterPlotSeries*> *scatterPlotSeries;
  bool autoBoundaries;
  double xMax;
  double xMin;
  double yMax;
  double yMin;
  bool autoPadding;
  double xPadding;
  double yPadding;
  std::vector<wchar_t> *xLabel;
  std::vector<wchar_t> *yLabel;
  std::vector<wchar_t> *title;
  bool showGrid;
  RGBA *gridColor;
  bool xAxisAuto;
  bool xAxisTop;
  bool xAxisBottom;
  bool yAxisAuto;
  bool yAxisLeft;
  bool yAxisRight;
  double width;
  double height;
};

struct BarPlotSeries{
  std::vector<double> *ys;
  RGBA *color;
};

struct BarPlotSettings{
  double width;
  double height;
  bool autoBoundaries;
  double yMax;
  double yMin;
  bool autoPadding;
  double xPadding;
  double yPadding;
  std::vector<wchar_t> *title;
  bool showGrid;
  RGBA *gridColor;
  std::vector<BarPlotSeries*> *barPlotSeries;
  std::vector<wchar_t> *yLabel;
  bool autoColor;
  bool grayscaleAutoColor;
  bool autoSpacing;
  double groupSeparation;
  double barSeparation;
  bool autoLabels;
  std::vector<StringReference*> *xLabels;
  bool barBorder;
};

struct RGBA{
  double r;
  double g;
  double b;
  double a;
};

struct RGBABitmap{
  std::vector<RGBA*> *y;
};

struct RGBABitmapImage{
  std::vector<RGBABitmap*> *x;
};

struct BooleanArrayReference{
  std::vector<bool> *booleanArray;
};

struct BooleanReference{
  bool booleanValue;
};

struct CharacterReference{
  wchar_t characterValue;
};

struct NumberArrayReference{
  std::vector<double> *numberArray;
};

struct NumberReference{
  double numberValue;
};

struct StringArrayReference{
  std::vector<StringReference*> *stringArray;
};

struct StringReference{
  std::vector<wchar_t> *string;
};

struct Chunk{
  double length;
  std::vector<wchar_t> *type;
  std::vector<double> *data;
  double crc;
};

struct IHDR{
  double Width;
  double Height;
  double BitDepth;
  double ColourType;
  double CompressionMethod;
  double FilterMethod;
  double InterlaceMethod;
};

struct PHYS{
  double pixelsPerMeter;
};

struct PNGImage{
  std::vector<double> *signature;
  IHDR *ihdr;
  ZLIBStruct *zlibStruct;
  bool physPresent;
  PHYS *phys;
};

struct ZLIBStruct{
  double CMF;
  double CM;
  double CINFO;
  double FLG;
  double FCHECK;
  double FDICT;
  double FLEVEL;
  std::vector<double> *CompressedDataBlocks;
  double Adler32CheckValue;
};

struct LinkedListNodeStrings{
  bool end;
  std::vector<wchar_t> *value;
  LinkedListNodeStrings *next;
};

struct LinkedListStrings{
  LinkedListNodeStrings *first;
  LinkedListNodeStrings *last;
};

struct LinkedListNodeNumbers{
  LinkedListNodeNumbers *next;
  bool end;
  double value;
};

struct LinkedListNumbers{
  LinkedListNodeNumbers *first;
  LinkedListNodeNumbers *last;
};

struct LinkedListCharacters{
  LinkedListNodeCharacters *first;
  LinkedListNodeCharacters *last;
};

struct LinkedListNodeCharacters{
  bool end;
  wchar_t value;
  LinkedListNodeCharacters *next;
};

struct DynamicArrayNumbers{
  std::vector<double> *array;
  double length;
};

bool CropLineWithinBoundary(NumberReference *x1Ref, NumberReference *y1Ref, NumberReference *x2Ref, NumberReference *y2Ref, double xMin, double xMax, double yMin, double yMax);
double IncrementFromCoordinates(double x1, double y1, double x2, double y2);
double InterceptFromCoordinates(double x1, double y1, double x2, double y2);

std::vector<RGBA*> *Get8HighContrastColors();

void DrawFilledRectangleWithBorder(RGBABitmapImage *image, double x, double y, double w, double h, RGBA *borderColor, RGBA *fillColor);
RGBABitmapImageReference *CreateRGBABitmapImageReference();

bool RectanglesOverlap(Rectangle *r1, Rectangle *r2);
Rectangle *CreateRectangle(double x1, double y1, double x2, double y2);
void CopyRectangleValues(Rectangle *rd, Rectangle *rs);

void DrawXLabelsForPriority(double p, double xMin, double oy, double xMax, double xPixelMin, double xPixelMax, NumberReference *nextRectangle, RGBA *gridLabelColor, RGBABitmapImage *canvas, std::vector<double> *xGridPositions, StringArrayReference *xLabels, NumberArrayReference *xLabelPriorities, std::vector<Rectangle*> *occupied, bool textOnBottom);
void DrawYLabelsForPriority(double p, double yMin, double ox, double yMax, double yPixelMin, double yPixelMax, NumberReference *nextRectangle, RGBA *gridLabelColor, RGBABitmapImage *canvas, std::vector<double> *yGridPositions, StringArrayReference *yLabels, NumberArrayReference *yLabelPriorities, std::vector<Rectangle*> *occupied, bool textOnLeft);
std::vector<double> *ComputeGridLinePositions(double cMin, double cMax, StringArrayReference *labels, NumberArrayReference *priorities);
double MapYCoordinate(double y, double yMin, double yMax, double yPixelMin, double yPixelMax);
double MapXCoordinate(double x, double xMin, double xMax, double xPixelMin, double xPixelMax);
double MapXCoordinateAutoSettings(double x, RGBABitmapImage *image, std::vector<double> *xs);
double MapYCoordinateAutoSettings(double y, RGBABitmapImage *image, std::vector<double> *ys);
double MapXCoordinateBasedOnSettings(double x, ScatterPlotSettings *settings);
double MapYCoordinateBasedOnSettings(double y, ScatterPlotSettings *settings);
double GetDefaultPaddingPercentage();

void DrawText(RGBABitmapImage *canvas, double x, double y, std::vector<wchar_t> *text, RGBA *color);
void DrawTextUpwards(RGBABitmapImage *canvas, double x, double y, std::vector<wchar_t> *text, RGBA *color);

ScatterPlotSettings *GetDefaultScatterPlotSettings();
ScatterPlotSeries *GetDefaultScatterPlotSeriesSettings();
bool DrawScatterPlot(RGBABitmapImageReference *canvasReference, double width, double height, std::vector<double> *xs, std::vector<double> *ys, StringReference *errorMessage);
bool DrawScatterPlotFromSettings(RGBABitmapImageReference *canvasReference, ScatterPlotSettings *settings, StringReference *errorMessage);
void ComputeBoundariesBasedOnSettings(ScatterPlotSettings *settings, Rectangle *boundaries);
bool ScatterPlotFromSettingsValid(ScatterPlotSettings *settings, StringReference *errorMessage);

BarPlotSettings *GetDefaultBarPlotSettings();
BarPlotSeries *GetDefaultBarPlotSeriesSettings();
RGBABitmapImage *DrawBarPlotNoErrorCheck(double width, double height, std::vector<double> *ys);
bool DrawBarPlot(RGBABitmapImageReference *canvasReference, double width, double height, std::vector<double> *ys, StringReference *errorMessage);
bool DrawBarPlotFromSettings(RGBABitmapImageReference *canvasReference, BarPlotSettings *settings, StringReference *errorMessage);
bool BarPlotSettingsIsValid(BarPlotSettings *settings, StringReference *errorMessage);

double GetMinimum(std::vector<double> *data);
double GetMaximum(std::vector<double> *data);

double RoundToDigits(double element, double digitsAfterPoint);

double test();
void TestMapping(NumberReference *failures);
void TestMapping2(NumberReference *failures);
void ExampleRegression(RGBABitmapImageReference *image);
void BarPlotExample(RGBABitmapImageReference *imageReference);

RGBA *GetBlack();
RGBA *GetWhite();
RGBA *GetTransparent();
RGBA *GetGray(double percentage);
RGBA *CreateRGBColor(double r, double g, double b);
RGBA *CreateRGBAColor(double r, double g, double b, double a);

RGBABitmapImage *CreateImage(double w, double h, RGBA *color);
void DeleteImage(RGBABitmapImage *image);
double ImageWidth(RGBABitmapImage *image);
double ImageHeight(RGBABitmapImage *image);
void SetPixel(RGBABitmapImage *image, double x, double y, RGBA *color);
void DrawPixel(RGBABitmapImage *image, double x, double y, RGBA *color);
double CombineAlpha(double as, double ad);
double AlphaBlend(double cs, double as, double cd, double ad, double ao);
void DrawHorizontalLine1px(RGBABitmapImage *image, double x, double y, double length, RGBA *color);
void DrawVerticalLine1px(RGBABitmapImage *image, double x, double y, double height, RGBA *color);
void DrawRectangle1px(RGBABitmapImage *image, double x, double y, double width, double height, RGBA *color);
void DrawImageOnImage(RGBABitmapImage *dst, RGBABitmapImage *src, double topx, double topy);
void DrawLine1px(RGBABitmapImage *image, double x0, double y0, double x1, double y1, RGBA *color);
void XiaolinWusLineAlgorithm(RGBABitmapImage *image, double x0, double y0, double x1, double y1, RGBA *color);
double OneMinusFractionalPart(double x);
double FractionalPart(double x);
RGBA *SetBrightness(RGBA *color, double newBrightness);
void DrawQuadraticBezierCurve(RGBABitmapImage *image, double x0, double y0, double cx, double cy, double x1, double y1, RGBA *color);
void QuadraticBezierPoint(double x0, double y0, double cx, double cy, double x1, double y1, double t, NumberReference *x, NumberReference *y);
void DrawCubicBezierCurve(RGBABitmapImage *image, double x0, double y0, double c0x, double c0y, double c1x, double c1y, double x1, double y1, RGBA *color);
void CubicBezierPoint(double x0, double y0, double c0x, double c0y, double c1x, double c1y, double x1, double y1, double t, NumberReference *x, NumberReference *y);
RGBABitmapImage *CopyImage(RGBABitmapImage *image);
RGBA *GetImagePixel(RGBABitmapImage *image, double x, double y);
void HorizontalFlip(RGBABitmapImage *img);
void DrawFilledRectangle(RGBABitmapImage *image, double x, double y, double w, double h, RGBA *color);
RGBABitmapImage *RotateAntiClockwise90Degrees(RGBABitmapImage *image);
void DrawCircle(RGBABitmapImage *canvas, double xCenter, double yCenter, double radius, RGBA *color);
void BresenhamsCircleDrawingAlgorithm(RGBABitmapImage *canvas, double xCenter, double yCenter, double radius, RGBA *color);
void DrawCircleMidpointAlgorithm(RGBABitmapImage *canvas, double xCenter, double yCenter, double radius, RGBA *color);
void DrawCircleBasicAlgorithm(RGBABitmapImage *canvas, double xCenter, double yCenter, double radius, RGBA *color);
void DrawFilledCircle(RGBABitmapImage *canvas, double x, double y, double r, RGBA *color);
void DrawFilledCircleMidpointAlgorithm(RGBABitmapImage *canvas, double xCenter, double yCenter, double radius, RGBA *color);
void DrawFilledCircleBasicAlgorithm(RGBABitmapImage *canvas, double xCenter, double yCenter, double radius, RGBA *color);
void DrawTriangle(RGBABitmapImage *canvas, double xCenter, double yCenter, double height, RGBA *color);
void DrawFilledTriangle(RGBABitmapImage *canvas, double xCenter, double yCenter, double height, RGBA *color);
void DrawLine(RGBABitmapImage *canvas, double x1, double y1, double x2, double y2, double thickness, RGBA *color);
void DrawLineBresenhamsAlgorithmThick(RGBABitmapImage *canvas, double x1, double y1, double x2, double y2, double thickness, RGBA *color);
void DrawLineBresenhamsAlgorithm(RGBABitmapImage *canvas, double x1, double y1, double x2, double y2, RGBA *color);
void DrawLineBresenhamsAlgorithmThickPatterned(RGBABitmapImage *canvas, double x1, double y1, double x2, double y2, double thickness, std::vector<bool> *pattern, NumberReference *offset, RGBA *color);

std::vector<bool> *GetLinePattern5();
std::vector<bool> *GetLinePattern4();
std::vector<bool> *GetLinePattern3();
std::vector<bool> *GetLinePattern2();
std::vector<bool> *GetLinePattern1();

RGBABitmapImage *Blur(RGBABitmapImage *src, double pixels);
RGBA *CreateBlurForPoint(RGBABitmapImage *src, double x, double y, double pixels);

std::vector<wchar_t> *CreateStringScientificNotationDecimalFromNumber(double decimal);
std::vector<wchar_t> *CreateStringScientificNotationDecimalFromNumber15d2e(double decimal);
std::vector<wchar_t> *CreateStringScientificNotationDecimalFromNumberAllOptions(double decimal, bool complete);
std::vector<wchar_t> *CreateStringDecimalFromNumber(double decimal);
bool CreateStringFromNumberWithCheck(double decimal, double base, StringReference *stringReference);
double GetMaximumDigitsForBase(double base);
double GetFirstDigitPosition(double decimal, double base);
bool GetSingleDigitCharacterFromNumberWithCheck(double c, double base, CharacterReference *characterReference);
std::vector<wchar_t> *GetDigitCharacterTable();

bool CreateNumberFromDecimalStringWithCheck(std::vector<wchar_t> *string, NumberReference *decimalReference, StringReference *errorMessage);
double CreateNumberFromDecimalString(std::vector<wchar_t> *string);
bool CreateNumberFromStringWithCheck(std::vector<wchar_t> *string, double base, NumberReference *numberReference, StringReference *errorMessage);
double CreateNumberFromParts(double base, bool numberIsPositive, std::vector<double> *beforePoint, std::vector<double> *afterPoint, bool exponentIsPositive, std::vector<double> *exponent);
bool ExtractPartsFromNumberString(std::vector<wchar_t> *n, double base, BooleanReference *numberIsPositive, NumberArrayReference *beforePoint, NumberArrayReference *afterPoint, BooleanReference *exponentIsPositive, NumberArrayReference *exponent, StringReference *errorMessages);
bool ExtractPartsFromNumberStringFromSign(std::vector<wchar_t> *n, double base, double i, NumberArrayReference *beforePoint, NumberArrayReference *afterPoint, BooleanReference *exponentIsPositive, NumberArrayReference *exponent, StringReference *errorMessages);
bool ExtractPartsFromNumberStringFromPointOrExponent(std::vector<wchar_t> *n, double base, double i, NumberArrayReference *afterPoint, BooleanReference *exponentIsPositive, NumberArrayReference *exponent, StringReference *errorMessages);
bool ExtractPartsFromNumberStringFromExponent(std::vector<wchar_t> *n, double base, double i, BooleanReference *exponentIsPositive, NumberArrayReference *exponent, StringReference *errorMessages);
double GetNumberFromNumberCharacterForBase(wchar_t c, double base);
bool CharacterIsNumberCharacterInBase(wchar_t c, double base);
std::vector<double> *StringToNumberArray(std::vector<wchar_t> *str);
bool StringToNumberArrayWithCheck(std::vector<wchar_t> *str, NumberArrayReference *numberArrayReference, StringReference *errorMessage);

double Negate(double x);
double Positive(double x);
double Factorial(double x);
double Round(double x);
double BankersRound(double x);
double Ceil(double x);
double Floor(double x);
double Truncate(double x);
double Absolute(double x);
double Logarithm(double x);
double NaturalLogarithm(double x);
double Sin(double x);
double Cos(double x);
double Tan(double x);
double Asin(double x);
double Acos(double x);
double Atan(double x);
double Atan2(double y, double x);
double Squareroot(double x);
double Exp(double x);
bool DivisibleBy(double a, double b);
double Combinations(double n, double k);
double Permutations(double n, double k);
bool EpsilonCompare(double a, double b, double epsilon);
double GreatestCommonDivisor(double a, double b);
double GCDWithSubtraction(double a, double b);
bool IsInteger(double a);
bool GreatestCommonDivisorWithCheck(double a, double b, NumberReference *gcdReference);
double LeastCommonMultiple(double a, double b);
double Sign(double a);
double Max(double a, double b);
double Min(double a, double b);
double Power(double a, double b);
double Gamma(double x);
double LogGamma(double x);
double LanczosApproximation(double z);
double Beta(double x, double y);
double Sinh(double x);
double Cosh(double x);
double Tanh(double x);
double Cot(double x);
double Sec(double x);
double Csc(double x);
double Coth(double x);
double Sech(double x);
double Csch(double x);
double Error(double x);
double ErrorInverse(double x);
double FallingFactorial(double x, double n);
double RisingFactorial(double x, double n);
double Hypergeometric(double a, double b, double c, double z, double maxIterations, double precision);
double HypergeometricDirect(double a, double b, double c, double z, double maxIterations, double precision);
double BernouilliNumber(double n);
double AkiyamaTanigawaAlgorithm(double n);

std::vector<double> *aStringToNumberArray(std::vector<wchar_t> *string);
std::vector<wchar_t> *aNumberArrayToString(std::vector<double> *array);
bool aNumberArraysEqual(std::vector<double> *a, std::vector<double> *b);
bool aBooleanArraysEqual(std::vector<bool> *a, std::vector<bool> *b);
bool aStringsEqual(std::vector<wchar_t> *a, std::vector<wchar_t> *b);
void aFillNumberArray(std::vector<double> *a, double value);
void aFillString(std::vector<wchar_t> *a, wchar_t value);
void aFillBooleanArray(std::vector<bool> *a, bool value);
bool aFillNumberArrayRange(std::vector<double> *a, double value, double from, double to);
bool aFillBooleanArrayRange(std::vector<bool> *a, bool value, double from, double to);
bool aFillStringRange(std::vector<wchar_t> *a, wchar_t value, double from, double to);
std::vector<double> *aCopyNumberArray(std::vector<double> *a);
std::vector<bool> *aCopyBooleanArray(std::vector<bool> *a);
std::vector<wchar_t> *aCopyString(std::vector<wchar_t> *a);
bool aCopyNumberArrayRange(std::vector<double> *a, double from, double to, NumberArrayReference *copyReference);
bool aCopyBooleanArrayRange(std::vector<bool> *a, double from, double to, BooleanArrayReference *copyReference);
bool aCopyStringRange(std::vector<wchar_t> *a, double from, double to, StringReference *copyReference);
bool aIsLastElement(double length, double index);
std::vector<double> *aCreateNumberArray(double length, double value);
std::vector<bool> *aCreateBooleanArray(double length, bool value);
std::vector<wchar_t> *aCreateString(double length, wchar_t value);
void aSwapElementsOfNumberArray(std::vector<double> *A, double ai, double bi);
void aSwapElementsOfStringArray(StringArrayReference *A, double ai, double bi);
void aReverseNumberArray(std::vector<double> *array);


BooleanReference *CreateBooleanReference(bool value);
BooleanArrayReference *CreateBooleanArrayReference(std::vector<bool> *value);
BooleanArrayReference *CreateBooleanArrayReferenceLengthValue(double length, bool value);
void FreeBooleanArrayReference(BooleanArrayReference *booleanArrayReference);
CharacterReference *CreateCharacterReference(wchar_t value);
NumberReference *CreateNumberReference(double value);
NumberArrayReference *CreateNumberArrayReference(std::vector<double> *value);
NumberArrayReference *CreateNumberArrayReferenceLengthValue(double length, double value);
void FreeNumberArrayReference(NumberArrayReference *numberArrayReference);
StringReference *CreateStringReference(std::vector<wchar_t> *value);
StringReference *CreateStringReferenceLengthValue(double length, wchar_t value);
void FreeStringReference(StringReference *stringReference);
StringArrayReference *CreateStringArrayReference(std::vector<StringReference*> *strings);
StringArrayReference *CreateStringArrayReferenceLengthValue(double length, std::vector<wchar_t> *value);
void FreeStringArrayReference(StringArrayReference *stringArrayReference);

std::vector<wchar_t> *DigitDataBase16();
void DrawDigitCharacter(RGBABitmapImage *image, double topx, double topy, double digit);

std::vector<wchar_t> *GetPixelFontData();
void DrawAsciiCharacter(RGBABitmapImage *image, double topx, double topy, wchar_t a, RGBA *color);
double GetTextWidth(std::vector<wchar_t> *text);
double GetTextHeight(std::vector<wchar_t> *text);

void AssertFalse(bool b, NumberReference *failures);
void AssertTrue(bool b, NumberReference *failures);
void AssertEquals(double a, double b, NumberReference *failures);
void AssertBooleansEqual(bool a, bool b, NumberReference *failures);
void AssertCharactersEqual(wchar_t a, wchar_t b, NumberReference *failures);
void AssertStringEquals(std::vector<wchar_t> *a, std::vector<wchar_t> *b, NumberReference *failures);
void AssertNumberArraysEqual(std::vector<double> *a, std::vector<double> *b, NumberReference *failures);
void AssertBooleanArraysEqual(std::vector<bool> *a, std::vector<bool> *b, NumberReference *failures);
void AssertStringArraysEqual(std::vector<StringReference*> *a, std::vector<StringReference*> *b, NumberReference *failures);

std::vector<double> *ConvertToPNG(RGBABitmapImage *image);
std::vector<double> *ConvertToPNGGrayscale(RGBABitmapImage *image);
PHYS *PysicsHeader(double pixelsPerMeter);
std::vector<double> *ConvertToPNGWithOptions(RGBABitmapImage *image, double colorType, bool setPhys, double pixelsPerMeter, double compressionLevel);
std::vector<double> *PNGSerializeChunks(PNGImage *png);
double PNGIDATLength(PNGImage *png);
double PNGHeaderLength();
std::vector<double> *GetPNGColorData(RGBABitmapImage *image);
std::vector<double> *GetPNGColorDataGreyscale(RGBABitmapImage *image);
IHDR *PNGHeader(RGBABitmapImage *image, double colortype);
std::vector<double> *PNGSignature();
std::vector<double> *PNGReadDataChunks(std::vector<Chunk*> *cs);
bool PNGReadHeader(RGBABitmapImage *image, std::vector<Chunk*> *cs, StringReference *errorMessages);
std::vector<Chunk*> *PNGReadChunks(std::vector<double> *data, NumberReference *position);
Chunk *PNGReadChunk(std::vector<double> *data, NumberReference *position);

void WriteStringToStingStream(std::vector<wchar_t> *stream, NumberReference *index, std::vector<wchar_t> *src);
void WriteCharacterToStingStream(std::vector<wchar_t> *stream, NumberReference *index, wchar_t src);
void WriteBooleanToStingStream(std::vector<wchar_t> *stream, NumberReference *index, bool src);

bool SubstringWithCheck(std::vector<wchar_t> *string, double from, double to, StringReference *stringReference);
std::vector<wchar_t> *Substring(std::vector<wchar_t> *string, double from, double to);
std::vector<wchar_t> *AppendString(std::vector<wchar_t> *s1, std::vector<wchar_t> *s2);
std::vector<wchar_t> *ConcatenateString(std::vector<wchar_t> *s1, std::vector<wchar_t> *s2);
std::vector<wchar_t> *AppendCharacter(std::vector<wchar_t> *string, wchar_t c);
std::vector<wchar_t> *ConcatenateCharacter(std::vector<wchar_t> *string, wchar_t c);
std::vector<StringReference*> *SplitByCharacter(std::vector<wchar_t> *toSplit, wchar_t splitBy);
bool IndexOfCharacter(std::vector<wchar_t> *string, wchar_t character, NumberReference *indexReference);
bool SubstringEqualsWithCheck(std::vector<wchar_t> *string, double from, std::vector<wchar_t> *substring, BooleanReference *equalsReference);
bool SubstringEquals(std::vector<wchar_t> *string, double from, std::vector<wchar_t> *substring);
bool IndexOfString(std::vector<wchar_t> *string, std::vector<wchar_t> *substring, NumberReference *indexReference);
bool ContainsCharacter(std::vector<wchar_t> *string, wchar_t character);
bool ContainsString(std::vector<wchar_t> *string, std::vector<wchar_t> *substring);
void ToUpperCase(std::vector<wchar_t> *string);
void ToLowerCase(std::vector<wchar_t> *string);
bool EqualsIgnoreCase(std::vector<wchar_t> *a, std::vector<wchar_t> *b);
std::vector<wchar_t> *ReplaceString(std::vector<wchar_t> *string, std::vector<wchar_t> *toReplace, std::vector<wchar_t> *replaceWith);
std::vector<wchar_t> *ReplaceCharacter(std::vector<wchar_t> *string, wchar_t toReplace, wchar_t replaceWith);
std::vector<wchar_t> *Trim(std::vector<wchar_t> *string);
bool StartsWith(std::vector<wchar_t> *string, std::vector<wchar_t> *start);
bool EndsWith(std::vector<wchar_t> *string, std::vector<wchar_t> *end);
std::vector<StringReference*> *SplitByString(std::vector<wchar_t> *toSplit, std::vector<wchar_t> *splitBy);
bool StringIsBefore(std::vector<wchar_t> *a, std::vector<wchar_t> *b);

std::vector<double> *ReadXbytes(std::vector<double> *data, NumberReference *position, double length);
double Read4bytesBE(std::vector<double> *data, NumberReference *position);
double Read2bytesBE(std::vector<double> *data, NumberReference *position);
double ReadByte(std::vector<double> *data, NumberReference *position);
double Read4bytesLE(std::vector<double> *data, NumberReference *position);
void WriteByte(std::vector<double> *data, double b, NumberReference *position);
void Write2BytesLE(std::vector<double> *data, double b, NumberReference *position);
void Write4BytesLE(std::vector<double> *data, double b, NumberReference *position);
void Write2BytesBE(std::vector<double> *data, double b, NumberReference *position);
void Write4BytesBE(std::vector<double> *data, double b, NumberReference *position);
void WriteStringBytes(std::vector<double> *data, std::vector<wchar_t> *cs, NumberReference *position);

std::vector<double> *MakeCRC32Table();
double UpdateCRC32(double crc, std::vector<double> *buf, std::vector<double> *crc_table);
double CalculateCRC32(std::vector<double> *buf);
double CRC32OfInterval(std::vector<double> *data, double from, double length);

ZLIBStruct *ZLibCompressNoCompression(std::vector<double> *data);
ZLIBStruct *ZLibCompressStaticHuffman(std::vector<double> *data, double level);

std::vector<double> *AddNumber(std::vector<double> *list, double a);
void AddNumberRef(NumberArrayReference *list, double i);
std::vector<double> *RemoveNumber(std::vector<double> *list, double n);
double GetNumberRef(NumberArrayReference *list, double i);
void RemoveNumberRef(NumberArrayReference *list, double i);

std::vector<StringReference*> *AddString(std::vector<StringReference*> *list, StringReference *a);
void AddStringRef(StringArrayReference *list, StringReference *i);
std::vector<StringReference*> *RemoveString(std::vector<StringReference*> *list, double n);
StringReference *GetStringRef(StringArrayReference *list, double i);
void RemoveStringRef(StringArrayReference *list, double i);

std::vector<bool> *AddBoolean(std::vector<bool> *list, bool a);
void AddBooleanRef(BooleanArrayReference *list, bool i);
std::vector<bool> *RemoveBoolean(std::vector<bool> *list, double n);
bool GetBooleanRef(BooleanArrayReference *list, double i);
void RemoveDecimalRef(BooleanArrayReference *list, double i);


LinkedListStrings *CreateLinkedListString();
void LinkedListAddString(LinkedListStrings *ll, std::vector<wchar_t> *value);
std::vector<StringReference*> *LinkedListStringsToArray(LinkedListStrings *ll);
double LinkedListStringsLength(LinkedListStrings *ll);
void FreeLinkedListString(LinkedListStrings *ll);


LinkedListNumbers *CreateLinkedListNumbers();
std::vector<LinkedListNumbers*> *CreateLinkedListNumbersArray(double length);
void LinkedListAddNumber(LinkedListNumbers *ll, double value);
double LinkedListNumbersLength(LinkedListNumbers *ll);
double LinkedListNumbersIndex(LinkedListNumbers *ll, double index);
void LinkedListInsertNumber(LinkedListNumbers *ll, double index, double value);
void LinkedListSet(LinkedListNumbers *ll, double index, double value);
void LinkedListRemoveNumber(LinkedListNumbers *ll, double index);
void FreeLinkedListNumbers(LinkedListNumbers *ll);
void FreeLinkedListNumbersArray(std::vector<LinkedListNumbers*> *lls);
std::vector<double> *LinkedListNumbersToArray(LinkedListNumbers *ll);
LinkedListNumbers *ArrayToLinkedListNumbers(std::vector<double> *array);
bool LinkedListNumbersEqual(LinkedListNumbers *a, LinkedListNumbers *b);

LinkedListCharacters *CreateLinkedListCharacter();
void LinkedListAddCharacter(LinkedListCharacters *ll, wchar_t value);
std::vector<wchar_t> *LinkedListCharactersToArray(LinkedListCharacters *ll);
double LinkedListCharactersLength(LinkedListCharacters *ll);
void FreeLinkedListCharacter(LinkedListCharacters *ll);



DynamicArrayNumbers *CreateDynamicArrayNumbers();
DynamicArrayNumbers *CreateDynamicArrayNumbersWithInitialCapacity(double capacity);
void DynamicArrayAddNumber(DynamicArrayNumbers *da, double value);
void DynamicArrayNumbersIncreaseSize(DynamicArrayNumbers *da);
bool DynamicArrayNumbersDecreaseSizeNecessary(DynamicArrayNumbers *da);
void DynamicArrayNumbersDecreaseSize(DynamicArrayNumbers *da);
double DynamicArrayNumbersIndex(DynamicArrayNumbers *da, double index);
double DynamicArrayNumbersLength(DynamicArrayNumbers *da);
void DynamicArrayInsertNumber(DynamicArrayNumbers *da, double index, double value);
void DynamicArraySet(DynamicArrayNumbers *da, double index, double value);
void DynamicArrayRemoveNumber(DynamicArrayNumbers *da, double index);
void FreeDynamicArrayNumbers(DynamicArrayNumbers *da);
std::vector<double> *DynamicArrayNumbersToArray(DynamicArrayNumbers *da);
DynamicArrayNumbers *ArrayToDynamicArrayNumbersWithOptimalSize(std::vector<double> *array);
DynamicArrayNumbers *ArrayToDynamicArrayNumbers(std::vector<double> *array);
bool DynamicArrayNumbersEqual(DynamicArrayNumbers *a, DynamicArrayNumbers *b);
LinkedListNumbers *DynamicArrayNumbersToLinkedList(DynamicArrayNumbers *da);
DynamicArrayNumbers *LinkedListToDynamicArrayNumbers(LinkedListNumbers *ll);

std::vector<wchar_t> *AddCharacter(std::vector<wchar_t> *list, wchar_t a);
void AddCharacterRef(StringReference *list, wchar_t i);
std::vector<wchar_t> *RemoveCharacter(std::vector<wchar_t> *list, double n);
wchar_t GetCharacterRef(StringReference *list, double i);
void RemoveCharacterRef(StringReference *list, double i);

wchar_t charToLowerCase(wchar_t character);
wchar_t charToUpperCase(wchar_t character);
bool charIsUpperCase(wchar_t character);
bool charIsLowerCase(wchar_t character);
bool charIsLetter(wchar_t character);
bool charIsNumber(wchar_t character);
bool charIsWhiteSpace(wchar_t character);
bool charIsSymbol(wchar_t character);
bool charCharacterIsBefore(wchar_t a, wchar_t b);

double And4Byte(double n1, double n2);
double And2Byte(double n1, double n2);
double AndByte(double n1, double n2);
double AndBytes(double n1, double n2, double bytes);
double Or4Byte(double n1, double n2);
double Or2Byte(double n1, double n2);
double OrByte(double n1, double n2);
double OrBytes(double n1, double n2, double bytes);
double Xor4Byte(double n1, double n2);
double Xor2Byte(double n1, double n2);
double XorByte(double n1, double n2);
double XorBytes(double n1, double n2, double bytes);
double Not4Byte(double b);
double Not2Byte(double b);
double NotByte(double b);
double NotBytes(double b, double length);
double ShiftLeft4Byte(double b, double amount);
double ShiftLeft2Byte(double b, double amount);
double ShiftLeftByte(double b, double amount);
double ShiftLeftBytes(double b, double amount, double length);
double ShiftRight4Byte(double b, double amount);
double ShiftRight2Byte(double b, double amount);
double ShiftRightByte(double b, double amount);
double ShiftRightBytes(double b, double amount, double length);

double ReadNextBit(std::vector<double> *data, NumberReference *nextbit);
double BitExtract(double b, double fromInc, double toInc);
double ReadBitRange(std::vector<double> *data, NumberReference *nextbit, double length);
void SkipToBoundary(NumberReference *nextbit);
double ReadNextByteBoundary(std::vector<double> *data, NumberReference *nextbit);
double Read2bytesByteBoundary(std::vector<double> *data, NumberReference *nextbit);

double ComputeAdler32(std::vector<double> *data);

std::vector<double> *DeflateDataStaticHuffman(std::vector<double> *data, double level);
void FindMatch(std::vector<double> *data, double pos, NumberReference *distanceReference, NumberReference *lengthReference, BooleanReference *match, double level);
std::vector<double> *GenerateBitReverseLookupTable(double bits);
double ReverseBits(double x, double bits);
std::vector<double> *DeflateDataNoCompression(std::vector<double> *data);
void GetDeflateStaticHuffmanCode(double b, NumberReference *code, NumberReference *length, std::vector<double> *bitReverseLookupTable);
void GetDeflateLengthCode(double length, NumberReference *code, NumberReference *lengthAddition, NumberReference *lengthAdditionLength);
void GetDeflateDistanceCode(double distance, NumberReference *code, NumberReference *distanceAdditionReference, NumberReference *distanceAdditionLengthReference, std::vector<double> *bitReverseLookupTable);
void AppendBitsToBytesLeft(std::vector<double> *bytes, NumberReference *nextbit, double data, double length);
void AppendBitsToBytesRight(std::vector<double> *bytes, NumberReference *nextbit, double data, double length);