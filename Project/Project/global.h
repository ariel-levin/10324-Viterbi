/******************************************
*******************************************
***		Ariel Levin						***
***		ariel.lvn89@gmail.com			***
***		http://about.me/ariel.levin		***
*******************************************
******************************************/

#ifndef __MAIN_H
#define __MAIN_H


#define NUM_OF_STATES	1000			// number of states, default: 1000
#define NUM_OF_OBSRV	30000			// number of time slices, default: 30000


#define FILE_PATH		"C:\\Users\\Ariel\\workspace\\afeka\\2015a\\10324-parallel\\Project\\Project\\Files\\"


/***************************************************************************
****************************	 SETTINGS		****************************
***************************************************************************/


// initialize consts
static const bool	TEST_VALUES = false;					// small scale 3x5 testing values
static const bool	WITH_FILES = true;						// working with input and output files
static const bool	GENERATE_ZEROES = false;				// when using random values: generate zeroes in observations
static const int	MAX_ZERO_RANGE = NUM_OF_OBSRV / 2;		// how often to generate zeros (if using random values)
static const bool	NORMALIZE_TRANS = false;				// normalize transition matrix

// how to calc consts
static const bool	WITH_LOGS = true;						// working with logs
static const bool	WITH_CUDA = true;						// working with cuda
static const bool	PRINT_OBSRV = false;					// print observation progress during calculation
															// (may slow the process)


/***************************************************************************
****************************	 STRUCTS		****************************
***************************************************************************/


typedef struct STATE
{
	float	prob;
	int		parent;

} STATE;

typedef struct MAX_STATE
{
	int		obsrv;
	int		state_num;
	STATE	state;

} MAX_STATE;


typedef enum { NORMAL, ZERO, LAST } flag;



#endif	//__MAIN_H

