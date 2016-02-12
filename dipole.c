/* Compile with: gcc -fpic -shared -lm -o libdip.so dipole.c */
#include <math.h>
/*  Returns field of a magnetic dipole in z direction in units of B0,
	where B0 is the field strength at one unit of distance.
	Calling program must multiply by B0.
	If the dipole moment is in -z direction, multiply output by -1. */
void dipole(double x, double y, double z, double* B)
{
	double r2 = x*x + y*y + z*z;
	double r = sqrt(r2);
	double r5 = r2*r2*r;
	B[0] = 3*x*z/r5;
	B[1] = 3*y*z/r5;
	B[2] = (3*z*z - r2)/r5;
}

void doubledipole(double x, double y, double z, double* B)
{
	double B1[3], B2[3];
	dipole(x,y,z,B1);
	dipole(x-20,y,z,B2);
	for(int i=0; i<3; i++)
		B[i] = B1[i] + B2[i];
}
