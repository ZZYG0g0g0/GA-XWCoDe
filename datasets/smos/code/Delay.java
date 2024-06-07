package smos.bean;

import java.io.Serializable;
import java.util.Date;

public class Delay implements Serializable {

	/**
	 * Class modeling a late entry of a student
	 * @author Nicola Pisanti
	 * @version 1.0 
	 */
	private static final long serialVersionUID = 78680884161960621L;

	private int idDelay;
	private int idUser;
	private Date dateDelay;
	private String timeDelay;
	private int academicYear;
	
	
	/**
	 * Method that returns the Delay id
     * @return an integer representing the delay id
	 */
	public int getIdDelay() {
		return idDelay;
	}
	/**
	 * Method setting the Delay id
     * @integer param representing the id to be set
	 */
	public void setIdDelay(int pIdDelay) {
		this.idDelay = pIdDelay;
	}
	/**
	 * Method that returns the id of the late student
     * @return an integer representing the student id
	 */
	public int getIdUser() {
		return idUser;
	}
	/**
	 * Method setting the student id relative to the delay
     * @param an integer representing the id to be set
	 */
	public void setIdUser(int pIdUser) {
		this.idUser = pIdUser;
	}
	/**
	 * Method that returns the date of delay
     * @return a string representing the date of the delay
	 */
	public Date getDateDelay() {
		return dateDelay;
	}
	/**
	 * Method setting the date of delay
     * @param a string representing the date of delay
	 */
	public void setDateDelay(Date pDateDelay) {
		this.dateDelay = pDateDelay;
	}
	/**
	 * Method that returns the student's time of entry
     * @return a string representing the time of entry of the late student
	 */
	public String getTimeDelay() {
		if (this.timeDelay.length() > 0){
			return timeDelay.substring(0, 5);
		} else {
			return this.timeDelay;
		}
	}
	/**
	 * Method setting the time of entry of the student
     * @param a string representing the input time to set
	 */
	public void setTimeDelay(String pTimeDelay) {
		this.timeDelay = pTimeDelay;
	}
	/**
	 * Method that returns the academic year relative to absence
     * @return an integer representing the year in which the academic year began
	 */
	public int getAcademicYear() {
		return academicYear;
	}
	/**
	 * Method setting the academic year relative to absence
     * @param an integer representing the academic year to be set
	 */
	public void setAcademicYear(int pAcademicYear) {
		this.academicYear = pAcademicYear;
	}
}
