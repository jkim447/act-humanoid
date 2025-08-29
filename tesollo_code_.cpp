Void SetTargetRight(double* q) // @params q = input data(degree){
	double dir[MOTOR_COUNT] = { 1,-1,1,1, -1,1,1,1, -1,1,1,1, -1,1,1,1, 1,-1,1,1 };
	double qd[MOTOR_COUNT];

	qd[0] = (58.5 - q[1]) * (PI / 180); // 58.5 = Initial angle of the Quantum metagloves
	qd[1] = (q[0] + 20) * (PI / 180); // 20 = Initial angle of the Quantum metagloves
	qd[2] = q[2] * (PI / 180);
	qd[3] = q[3] * (PI / 180);

	qd[4] = q[4] * (PI / 180);
	qd[5] = q[5] * (PI / 180);
	qd[6] = q[6] * (PI / 180);
	qd[7] = q[7] * (PI / 180);

	qd[8] = q[8] * (PI / 180);
	qd[9] = q[9] * (PI / 180);
	qd[10] = q[10] * (PI / 180);
	qd[11] = q[11] * (PI / 180);

	qd[12] = q[12] * (PI / 180);
	qd[13] = q[13] * (PI / 180);
	qd[14] = q[14] * (PI / 180);
	qd[15] = q[15] * (PI / 180);

	if (q[17] > 55 && q[18] > 25 && q[18] > 20) // Ratio variation of the pinky finger between bent and straight positions
	{
		qd[16] = abs(q[16]) * 2 * (PI / 180);
	}
	else
	{
		qd[16] = abs(q[16]) / 1.5 * (PI / 180);
	}
	qd[17] = q[16] * (PI / 180);
	qd[18] = q[17] * (PI / 180);
	qd[19] = (q[19]) * (PI / 180);

	for (int i = 0; i < 20; i++)
	{
		mQd[i] = qd[i] * mGripperCalibrationData[i] * dir[i]; //  mQd = DG-5F Target Joint Data
		switch (i)
		{
		case 4:
		case 8:
		case 12:
		case 17:
		case 16:
		{