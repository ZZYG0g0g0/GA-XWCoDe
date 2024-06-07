package smos.utility;

import smos.exception.EntityNotFoundException;
import smos.storage.connectionManagement.DBConnection;
import smos.storage.connectionManagement.exception.ConnectionException;

import java.util.Calendar;
import java.util.Date;
import java.util.GregorianCalendar;
import java.util.TimeZone;
import java.sql.*;
import java.text.DateFormat;

/**
 * This class contains a set of system management utilities.
 */
public class Utility {

	/**
	 * Constants
	 */
	public static final char SLASH = (char) 47;// character '/' in ascii

	/**
	 * 
	 */
	public static final char BACKSLASH = (char) 92;// character '/' in ascii

	/**
	 * 
	 */
	public static final String[] day = { "Monday","Tuesday","Wednesday"
			"Thursday","Friday","Saturday"};
	
	/**
	 * 
	 */
	public static final String[] validHour = { "08:00", "08:30", "09:00",
		"09:30", "10:00", "10:30", "11:00", "11:30", "12:00", "12:30",
		"13:00", "13:30", "14:00", "14:30", "15:00", "15:30", "16:00",
		"16:30", "17:00", "17:30", "18:00", "18:30", "19:00", "19:30",
		"20:00" };
	
	public static final String[] month = {"Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"};

	/**
	 * 
	 */
	public static final String[] requestState = { "carried out", "rejected",
			"accepted", "eliminated" };
	
	//Variables to set when Tomcat starts
	
	//Variables for the graphic aspect of the program
	private static String imageHeaderPath = "";
	private static String imageHeaderLogoPath = "";
	private static String imageLeftColumn = "";
	private static String textFooter = "";
	//smtp server for sending e-mails
	private static String serverSmtp = "smtp.tele2.it";
	//Minimum days away from the exam to subscribe or cancel a reservation
	private static int needDayBeforeExam = 0;
	//Maximum days away from the exam to sign up for a reservation
	private static int maxDayBeforeExam = 0;
	//pdf path for the storage path of the pdf files
	private static String pdfPath = "";
	// upload path for the file storage path for importing data
	private static String uploadPath = "";
	//Variables for connecting to the database
	private static String driverMySql = "com.mysql.jdbc.Driver";
	private static String fullPathDatabase = "jdbc:mysql://localhost:3306/smos";
	private static String userName = "root";
	private static String password = "";
	private static int maxPoolSize = 200;
	private static int waitTimeout = 120000;
	private static int activeTimeout = 600000;
	private static int poolTimeout = 27000000;	
	
	

	/**
	 * Returns, given a field and a table, the maximum value of the field in the
	 * table.
	 * 
	 * @param pField
	 *            The field of which we want to retrieve the maximum value.
	 * @param pTable
	 *            The table in which to search for the required information.
	 * @return Returns the maximum value, of the pField field, passed as
	 *         parameter, present in the pTable table, passed as a parameter.
	 * 
	 * @throws EntityNotFoundException
	 * @throws ConnectionException
	 * @throws SQLException
	 */
	synchronized static public int getMaxValue(String pField, String pTable)
			throws EntityNotFoundException, ConnectionException, SQLException {

		int value = 0;
		Connection connect = null;
		try {
			/*
			 * If the field and table have not been provided we return a
			 * error code
			 */
			if (pField.equals(""))
				throw new EntityNotFoundException();
			if (pTable.equals(""))
				throw new EntityNotFoundException();

			/*
			 * We prepare the SQL string to retrieve the information
			 * requests
			 */
			String tSql = "SELECT max(" + pField + ") as new_field FROM "
					+ pTable;

			// We get a connection to the database
			connect = DBConnection.getConnection();
			if (connect == null)
				throw new ConnectionException();

			// We send the Query to the DataBase
			ResultSet tRs = Utility.queryOperation(connect, tSql);

			if (tRs.next())
				value = tRs.getInt("new_field");
			else
				throw new EntityNotFoundException();

			return value;
		} finally {
			DBConnection.releaseConnection(connect);
		}
	}

	/**
	 * Converts a date to a formatted string for the database.
	 * 
	 * @param pDate
	 *            The date to be converted.
	 * @return Returns the formatted date for the database.
	 */
	static public String date2SQLString(java.util.Date pDate, boolean pHour) {
		
		TimeZone tz = TimeZone.getDefault();
		Calendar calendar = Calendar.getInstance(tz);
		calendar.setTime(pDate);
		
		
		int year = calendar.get(Calendar.YEAR);
		int month = calendar.get(Calendar.MONTH) + 1;
		int day = calendar.get(Calendar.DATE);
		
		String result = year + "-" + month + "-" + day ;
		
		int hour = calendar.get(Calendar.HOUR_OF_DAY);
		int min = calendar.get(Calendar.MINUTE);
		int sec = calendar.get(Calendar.SECOND);
		
		if (pHour){
			result = result + " " + hour + ":" + min + ":" + sec; 
		}
						

		return result;
	}

	/**
	 * Check if the string passed as a parameter is null.
	 * 
	 * @param pStr
	 *            The string to check.
	 * 
	 * @return <code> null </code> if the string is null Otherwise the string is
	 *         passed to the <code> Replace </code> method
	 * 
	 * @see #Replace
	 */
	static public String isNull(String pStr) {
		String tTmp;

		if (pStr == null) 
			tTmp = "null";

		else {
			pStr = ReplaceAllStrings(pStr, "'", "\\'");
			pStr = ReplaceAllStrings(pStr, "\"", "\\" + "\"");
			tTmp = "'" + pStr + "'";
		}
		return tTmp;
	}

	/**
	 * Checks if a given integer is null.
	 * 
	 * @param pInt
	 *            The whole data to be checked.
	 * @return Returns the data in string format if the data is different from zero,
	 *        otherwise the null string.
	 */
	static public String isNull(Integer pInt) {
		String tIntString;

		if (pInt == null)
			tIntString = "null";
		else
			tIntString = pInt.toString();
		return tIntString;
	}

	/**
	 * Check if a date is null.
	 * 
	 * @param pDate
	 *            the date to be checked.
	 * 
	 * @return Returns the string representing the date passed as input,
	 *         formatted for the database.
	 */
	static public String isNull(java.util.Date pDate) {

		return "'" + date2SQLString(pDate,false) + "'";
	}

	static public String clear(String pString){
		int start = 0;
		int start2 = 0;
		int end = 0;
		int end2 = 0;
		String result = "";
		String result2 = "";
			
		while (end >= 0){
		end = pString.indexOf("'", start);
		
		if (end >= 0){
		
			result= result +pString.substring(start, end);
			result = result + "\\'";
			start = end+1;
			
		} else 
			result = result + pString.substring(start);
		}
		
		while (end2 >= 0){
			end2 = result.indexOf("\n", start2);
			
			if (end2 >= 0){
				
				result2= result2 +result.substring(start2, end2-1);
				result2 = result2 + " ";
				start2 = end2+1;
				
			} else 
				result2 = result2 + result.substring(start2);
			}
		return(result2);	
	}
	
	/**
	 * Convert a String to Integer.
	 * 
	 * @param pStr
	 *            The String to convert.
	 * 
	 * @return The integer contained in the String <code> 0 </code> if it is not
	 *         conversion possible.
	 * 
	 */
	static public Integer String2Integer(String pStr) {
		Integer tInteger;

		if ((pStr == null) || (pStr.compareTo("") == 0))
			tInteger = new Integer(0);
		else
			try {
				tInteger = Integer.decode(pStr);
			} catch (Exception e) {
				tInteger = new Integer(0);
			}

		return tInteger;
	}

	/**
	 * Replace the characters' and \ with '' in the string passed as
	 * parameter.
	 * 
	 * @param pStr
	 *            The string to transform.
	 * 
	 * return The transformed string.
	 */
	static public String Replace(String pStr) {
		String tRis;

		tRis = pStr.replaceAll("\"", "'");

		tRis = tRis.replaceAll("'", "\\'");

		return tRis;
	}

	/**
	 * Replaces the 'character with the string "" in the string passed as
	 * parameter.
	 * 
	 * @param pStr
	 *            The string to transform.
	 * 
	 * @return The transformed string.
	 */
	static public String ReplaceQuote(String pStr) {
		String tRis;

		tRis = pStr.replaceAll("'", " ");
		return tRis;
	}

	/**
	 * Performs a database operation using an SQL string.
	 * 
	 * @param pConnect
	 *            The connection to the database.
	 * @param pSql
	 *            The SQL string.
	 * 
	 * @return The number of records involved in the operation.
	 * 
	 * @throws SQLException
	 */
	static public int executeOperation(Connection pConnect, String pSql)
			throws SQLException {
		Statement stmt = pConnect.createStatement();
		int tResult = stmt.executeUpdate(pSql);
		stmt.close();
		return tResult;
	}

	/**
	 * Executes an SQL Query on the database.
	 * 
	 * @param pConnect
	 *            The connection to the database.
	 * @param pSql
	 *            The SQL string.
	 * 
	 * @return The number of records involved in the operation.
	 * 
	 * @throws SQLException
	 */
	static public ResultSet queryOperation(Connection pConnect, String pSql)
			throws SQLException {

		Statement stmt = pConnect.createStatement();
		return stmt.executeQuery(pSql);
	}

	/**
	 * Converts Boolean data to integer.
	 * 
	 * @param pBol
	 *            The Boolean value to convert to an integer.
	 * 
	 * @return The integer value corresponding to the Boolean value passed as
	 *         parameter.
	 */
	static public int BooleanToInt(boolean pBol) {
		if (pBol == true)
			return 1;
		else
			return 0;
	}

	/**
	 * Converts a given integer to Boolean.
	 * 
	 * @param pInt
	 *            The integer value to convert to Boolean.
	 * 
	 * @return The boolean value corresponding to the integer value passed as
	 *         parameter.
	 */
	static public boolean IntToBoolean(int pInt) {
		if (pInt == 1)
			return true;
		else
			return false;
	}

	/**
	 * Returns the current date.
	 * 
	 * @return The current date.
	 */
	static public java.util.Date today() {
		Calendar calendar = Calendar.getInstance();
		java.util.Date creationDate = calendar.getTime();
		return creationDate;
		// java.util.Date creationDate = new java.util.Date();
		// Timestamp timeStamp = new Timestamp(creationDate.getTime());
		// return (java.util.Date) timeStamp;
	}

	/**
	 * Returns the date contained in the input string.
	 * 
	 * @param pDate
	 *            The string to parse on a date.
	 * 
	 * @return The converted date.
	 */
	static public java.util.Date String2Date(String pDate) {
		try {
			DateFormat dfDate = DateFormat.getDateInstance();
			java.util.Date tDate = dfDate.parse(pDate,
					new java.text.ParsePosition(0));
			java.sql.Timestamp timeStamp = new java.sql.Timestamp(tDate
					.getTime());

			return (java.util.Date) timeStamp;
		} catch (Exception e) {
			try {
				DateFormat dfDate = DateFormat.getDateInstance(
						DateFormat.SHORT, java.util.Locale.ITALY);
				java.util.Date tDate = dfDate.parse(pDate,
						new java.text.ParsePosition(0));
				java.sql.Timestamp timeStamp = new java.sql.Timestamp(tDate
						.getTime());

				return (java.util.Date) timeStamp;
			} catch (Exception e2) {
				return null;
			}
		}
	}

	/**
	 * Returns the input date in String format.
	 * 
	 * @param pDate
	 *            The Date to be converted.
	 * @param pHour
	 * 
	 * @return The converted date.
	 */
	static public String Date2String(java.util.Date pDate, boolean pHour) {
		try {
			DateFormat dfDate = DateFormat.getDateInstance(DateFormat.SHORT);
			DateFormat dfTime = DateFormat.getTimeInstance(DateFormat.SHORT);
			if (pHour)
				return dfDate.format(pDate) + " " + dfTime.format(pDate);
			else
				return dfDate.format(pDate);
		} catch (Exception e) {
			return "";
		}
	}

	/**
	 * @param sourceStr
	 * @param searchFor
	 * @param replaceWith
	 * @return the string corresponding to searchBuffer
	 */
	public static String ReplaceAll(String sourceStr, String searchFor,
			String replaceWith) {
		StringBuffer searchBuffer = new StringBuffer(sourceStr);
		int hits = 0;

		while (searchBuffer.toString().toUpperCase().indexOf(
				searchFor.toUpperCase(), hits) >= 0) {
			int newIndex = searchBuffer.toString().toUpperCase().indexOf(
					searchFor.toUpperCase(), hits);
			searchBuffer.replace(newIndex, newIndex + searchFor.length(),
					replaceWith);
			hits++;
		}

		return searchBuffer.toString();
	}

	// The previous function loops if I replace "\" with "\\"
	// then we use two variables the first is "consumed" at each occurrence
	// of the pattern
	/**
	 * @param sourceStr
	 * @param searchFor
	 * @param replaceWith
	 * @return newStringBuffer
	 */
	public static String ReplaceAllStrings(String sourceStr, String searchFor,
			String replaceWith) {
		StringBuffer searchBuffer = new StringBuffer(sourceStr);
		StringBuffer newStringBuffer = new StringBuffer("");

		while (searchBuffer.toString().toUpperCase().indexOf(
				searchFor.toUpperCase()) >= 0) {
			int newIndex = searchBuffer.toString().toUpperCase().indexOf(
					searchFor.toUpperCase());
			newStringBuffer.append(searchBuffer.substring(0, newIndex));
			newStringBuffer.append(replaceWith);
			searchBuffer = new StringBuffer(searchBuffer.substring(newIndex
					+ searchFor.length(), searchBuffer.length()));
		}

		newStringBuffer.append(searchBuffer);
		return newStringBuffer.toString();
	}

	/**
	 * @param sourceStr
	 * @param removeStr
	 */
	public static void RemoveAll(String sourceStr, String removeStr) {
		int nextOccurence;

		while (sourceStr.toString().toUpperCase().indexOf(
				removeStr.toUpperCase()) >= 0) {
			nextOccurence = sourceStr.toString().toUpperCase().indexOf(
					removeStr.toUpperCase());
			sourceStr = sourceStr.substring(0, nextOccurence)
					+ sourceStr.substring(nextOccurence + removeStr.length(),
							sourceStr.length());
		}
	}
	
	
	/**
	 * Calculate the distance in days between 2 past dates.
	 */
	@SuppressWarnings("deprecation")
	public static int daysBetween(Date today, Date reservationDate){
		int daysBetween = 0;
		long millisecBetween = 0;
		final int millisecInADay = 86400000;
		
		GregorianCalendar firstDate = new GregorianCalendar();
		GregorianCalendar secondDate = new GregorianCalendar();
		
		firstDate.set(today.getYear()+1900, today.getMonth(), today.getDate());
		secondDate.set(reservationDate.getYear()+1900, reservationDate.getMonth(), reservationDate.getDate());
		
		millisecBetween = secondDate.getTimeInMillis() - firstDate.getTimeInMillis();
		daysBetween = (int) (millisecBetween/millisecInADay);
		
		return (daysBetween);
	}

	/**
	 * @return The path set for the pdf.
	 */
	@SuppressWarnings("static-access")
	public static String getPdfPath() {
		return Utility.pdfPath;
	}

	/**
	 * @param pPdfPath The path to be set for the pdf.
	 */
	public static void setPdfPath(String pPdfPath) {
		Utility.pdfPath = pPdfPath;
	}
	
	/**
	 * @return The path set for the files used
	 * when importing data.
	 */
	@SuppressWarnings("static-access")
	public static String getUploadPath() {
		return Utility.uploadPath;
	}

	/**
	 * @param pUploadPath The path to set for the files used
	 * when importing data.
	 */
	public static void setUploadPath(String pUploadPath) {
		Utility.uploadPath = pUploadPath;
	}

	/**
	 * @return The smtp server to use for sending
	 * 		   automated e-mails.
	 */
	@SuppressWarnings("static-access")
	public static String getServerSmtp() {
		return Utility.serverSmtp;
	}
	
	/**
	 * @param pServerSmtp the smtp server to be set.
	 */
	public static void setServerSmtp(String pServerSmtp) {
		Utility.serverSmtp = pServerSmtp;
	}

	/**
	 * @return the activeTimeout
	 */
	public static int getActiveTimeout() {
		return Utility.activeTimeout;
	}

	/**
	 * @param activeTimeout the activeTimeout to set
	 */
	public static void setActiveTimeout(int pActiveTimeout) {
		Utility.activeTimeout = pActiveTimeout;
	}

	/**
	 * @return the driverMySql
	 */
	public static String getDriverMySql() {
		return Utility.driverMySql;
	}

	/**
	 * @param driverMySql the driverMySql to set
	 */
	public static void setDriverMySql(String pDriverMySql) {
		Utility.driverMySql = pDriverMySql;
	}

	/**
	 * @return the fullPathDatabase
	 */
	public static String getFullPathDatabase() {
		return Utility.fullPathDatabase;
	}

	/**
	 * @param fullPathDatabase the fullPathDatabase to set
	 */
	public static void setFullPathDatabase(String pFullPathDatabase) {
		Utility.fullPathDatabase = pFullPathDatabase;
	}

	/**
	 * @return the maxPoolSize
	 */
	public static int getMaxPoolSize() {
		return Utility.maxPoolSize;
	}

	/**
	 * @param maxPoolSize the maxPoolSize to set
	 */
	public static void setMaxPoolSize(int pMaxPoolSize) {
		Utility.maxPoolSize = pMaxPoolSize;
	}

	/**
	 * @return the password
	 */
	public static String getPassword() {
		return Utility.password;
	}

	/**
	 * @param password the password to set
	 */
	public static void setPassword(String pPassword) {
		Utility.password = pPassword;
	}


	/**
	 * @return the poolTimeout
	 */
	public static int getPoolTimeout() {
		return Utility.poolTimeout;
	}

	/**
	 * @param poolTimeout the poolTimeout to set
	 */
	public static void setPoolTimeout(int pPoolTimeout) {
		Utility.poolTimeout = pPoolTimeout;
	}

	/**
	 * @return the userName
	 */
	public static String getUserName() {
		return Utility.userName;
	}

	/**
	 * @param userName the userName to set
	 */
	public static void setUserName(String pUserName) {
		Utility.userName = pUserName;
	}

	/**
	 * @return the waitTimeout
	 */
	public static int getWaitTimeout() {
		return Utility.waitTimeout;
	}

	/**
	 * @param waitTimeout the waitTimeout to set
	 */
	public static void setWaitTimeout(int pWaitTimeout) {
		Utility.waitTimeout = pWaitTimeout;
	}

	/**
	 * @return the imageHeaderPath
	 */
	public static String getImageHeaderPath() {
		return Utility.imageHeaderPath;
	}

	/**
	 * @param imageHeaderPath the imageHeaderPath to set
	 */
	public static void setImageHeaderPath(String pImageHeaderPath) {
		Utility.imageHeaderPath = pImageHeaderPath;
	}

	/**
	 * @return the textFooter
	 */
	public static String getTextFooter() {
		return Utility.textFooter;
	}

	/**
	 * @param textFooter the textFooter to set
	 */
	public static void setTextFooter(String pTextFooter) {
		Utility.textFooter = pTextFooter;
	}

	/**
	 * @return the imageLeftColumn
	 */
	public static String getImageLeftColumn() {
		return Utility.imageLeftColumn;
	}

	/**
	 * @param imageLeftColumn the imageLeftColumn to set
	 */
	public static void setImageLeftColumn(String pImageLeftColumn) {
		Utility.imageLeftColumn = pImageLeftColumn;
	}
	
	public static String getImageLogoHeaderPath() {
		return Utility.imageHeaderLogoPath;
	}

	public static void setImageLogoHeaderPath(String pImageHeaderLogoPath) {
		Utility.imageHeaderLogoPath = pImageHeaderLogoPath;
	}

	/**
	 * @return the needDayBeforeExam
	 */
	public static int getNeedDayBeforeExam() {
		return Utility.needDayBeforeExam;
	}

	/**
	 * @param needDayBeforeExam the needDayBeforeExam to set
	 */
	public static void setNeedDayBeforeExam(int pNeedDayBeforeExam) {
		Utility.needDayBeforeExam = pNeedDayBeforeExam;
	}

	/**
	 * @return the maxDayBeforeExam
	 */
	public static int getMaxDayBeforeExam() {
		return Utility.maxDayBeforeExam;
	}

	/**
	 * @param maxDayBeforeExam the maxDayBeforeExam to set
	 */
	public static void setMaxDayBeforeExam(int pMaxDayBeforeExam) {
		Utility.maxDayBeforeExam = pMaxDayBeforeExam;
	}
	
	public static String getActualDate() {
		GregorianCalendar gc = new GregorianCalendar();

		String date="";
		int year=gc.get(GregorianCalendar.YEAR);
		
		int month=gc.get(GregorianCalendar.MONTH)+1;
		String months="";
		if(month<10){
			months="0"+month;
		}else{
			months= months+month;
		}
		
		int day=gc.get(GregorianCalendar.DAY_OF_MONTH);
		String days="";
		if(day<10){
			days="0"+day;
		}else{
			days= days+day;
		}
		
		date= date + days+"/"+months+"/"+year;
	
		return date;
		
	}
}
