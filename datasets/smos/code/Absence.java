package smos.bean;

import java.io.Serializable;
import java.util.Date;
public class Absence implements Serializable {

	/**
	 * Class that models the absence of a student
	 * @author Nicola Pisanti
	 * @version 1.0 
	 */
	private static final long serialVersionUID = -8396513309450121449L;
	
	private int idAbsence;
	private int idUser;
	private Date dateAbsence;
	private Integer idJustify;
	private int academicYear;
	
	public Absence (){
		
	}
	
	/**
	 * Method that returns the id of absence
	 * @return an integer representing the id of the absence
	 */
	public int getIdAbsence() {
		return idAbsence;
	}
	/**
	 * Method setting the id of absence
	 * @param an integer representing the id to be set
	 */
	public void setIdAbsence(int pIdAbsence) {
		this.idAbsence = pIdAbsence;
	}
	/**
	 * Method that returns the student id relative to the absence
	 * @return an integer representing the id of the absent student
	 */
	public int getIdUser() {
		return idUser;
	}
	/**
	 * Method setting student id relative to absence
	 * @param an integer representing the id to be set
	 */
	public void setIdUser(int pIdUser) {
		this.idUser = pIdUser;
	}
	/**
	 * Method that returns the date of absence
	 * @return a string representing the date of absence
	 */
	public Date getDateAbsence() {
		return dateAbsence;
	}
	/**
	 * Method setting the date of absence
	 * @param a string with the date to set
	 */
	public void setDateAbsence(Date pDateAbsence) {
		this.dateAbsence = pDateAbsence;
	}
	/**
	 * Method that returns the id of the justification relative to the absence
	 * @return an integer representing the id of the justification relative to the absence, or null if the absence was not justified
	 */
	public Integer getIdJustify() {
		
		return idJustify;
		
	}
	/**
	 * Method setting the id of the justification relative to the absence
	 * @param an integer representing the id of the justify to be set
	 */
	public void setIdJustify(Integer pIdJustify) {
		this.idJustify = pIdJustify;
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
